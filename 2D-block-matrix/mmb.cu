/******************************************************************************
 *
 * Distributed Multi-GPU Matrix Multiplication Benchmark using 2D matrix block decomposition (SUMMA) 
 *
 * Description:
 *   Distributed matrix multiplication benchmark for evaluating different
 *   communication libraries in multi-GPU systems.
 *
 *   Supported communication libraries. The three-character argument specifies the communication library used
 *   by the benchmark. For example:
 *
 *     MMM = MPI synchronous
 *     YYY = MPI asynchronous (nonblocking collectives + B/GEMM overlap)
 *     CCC = CUDA-Aware MPI synchronous
 *     XXX = CUDA-Aware MPI asynchronous (nonblocking collectives + B/GEMM overlap)
 *     NNN = NCCL
 *     WWW = NVSHMEM synchronous (full B GET before GEMM)
 *     SSS = NVSHMEM
 *
 * Compilation:
 *
 *   [murilo.boratto@sdumont]$ module load nvshmem/3.1.7_cuda-11.2_sequana
 * 
 *   [murilo.boratto@sdumont]$ module list
 * 
 *   Currently Loaded Modules:
 *       1) xpmem/2.6.5_sequana          3) gcc/9.3_sequana                       5) cuda/11.2_sequana             7) nvshmem/3.1.7_cuda-11.2_sequana
 *       2) ucx/1.13+cuda-11.2_sequana   4) openmpi/gnu/4.1.4+cuda-11.2_sequana   6) nccl/2.13_cuda-11.2_sequana
 *
 *   [murilo.boratto@sdumont]$ make
 *
 * Execution:
 *
 *   Example using one node with four GPUs and one MPI process per GPU:
 *
 *   [murilo.boratto@sdumont]$ mpirun -np 1 ./mmb 0 2048 MMM \
 *                                  : -np 1 ./mmb 1 2048 MMM \
 *                                  : -np 1 ./mmb 2 2048 MMM \
 *                                  : -np 1 ./mmb 3 2048 MMM
 *
 *   Arguments:
 *
 *   ./mmb <device_id> <matrix_size> <libraries>
 *
 *   where:
 *
 *     device_id    = CUDA GPU device assigned to the MPI process
 *     matrix_size  = matrix dimension (e.g., 2048)
 *     libraries    = communication library combination (MMM, YYY, CCC, XXX, NNN, SSS, WWW)
 *
 *   In the example above:
 *
 *     MPI Rank 0 -> GPU 0
 *     MPI Rank 1 -> GPU 1
 *     MPI Rank 2 -> GPU 2
 *     MPI Rank 3 -> GPU 3
 *
 *   Therefore, four MPI processes are launched, each associated with one
 *   NVIDIA GPU.
 *
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#define TILE_DIM 32

extern void ABMultiplyAccumulate(double *a, double *b, double *c, int m, int n, int k, int lda, int ldb, int ldc, int w);
extern void ABMultiplyAccumulateAsync(const double *a, const double *b, double *c, int m, int n, int k_total, int k_offset, int k_chunk, int lda, int ldb, int ldc, int w, cudaStream_t stream);
extern int validate_matrix_C(const double *d_C, int rows, int cols, double expected, double abs_tol, double rel_tol, int rank);

static void pack_block(const double *M, double *block, int N, int bs, int block_row, int block_col)
{
    for (int r = 0; r < bs; ++r)
        memcpy(block + (size_t)r * bs, M + (size_t)(block_row * bs + r) * N + block_col * bs, (size_t)bs * sizeof(double));
}

static void unpack_block(double *M, const double *block, int N, int bs, int block_row, int block_col)
{
    for (int r = 0; r < bs; ++r)
        memcpy(M + (size_t)(block_row * bs + r) * N + block_col * bs, block + (size_t)r * bs, (size_t)bs * sizeof(double));
}

static void pack_A_chunk(const double *Aij, double *dst, int bs, int k0, int kc)
{
    for (int r = 0; r < bs; ++r)
        memcpy(dst + (size_t)r * kc, Aij + (size_t)r * bs + k0, (size_t)kc * sizeof(double));
}

static void pack_B_chunk(const double *Bij, double *dst, int bs, int k0, int kc)
{
    memcpy(dst, Bij + (size_t)k0 * bs, (size_t)kc * bs * sizeof(double));
}

static void post_summa_chunk(int k, int k0, int kc, int slot, int proc_row, int proc_col, int bs, const double *h_Aij, const double *h_Bij, double *h_Achunk[2], double *h_Bchunk[2], MPI_Comm row_comm, MPI_Comm col_comm, MPI_Request reqA[2], MPI_Request reqB[2])
{
    const int a_count = bs * kc;
    const int b_count = kc * bs;

    if (proc_col == k)
        pack_A_chunk(h_Aij, h_Achunk[slot], bs, k0, kc);

    MPI_Ibcast(h_Achunk[slot], a_count, MPI_DOUBLE, k, row_comm, &reqA[slot]);

    if (proc_row == k)
        pack_B_chunk(h_Bij, h_Bchunk[slot], bs, k0, kc);

    MPI_Ibcast(h_Bchunk[slot], b_count, MPI_DOUBLE, k, col_comm, &reqB[slot]);
}

/*****************************************************************************************/

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, nranks, name_len;
    char name[MPI_MAX_PROCESSOR_NAME];
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Get_processor_name(name, &name_len);

    if (argc != 4) 
    {
        if (rank == 0)
            fprintf(stderr, "Usage: %s <device_id> <matrix_size> < MMM | YYY | CCC | XXX | NNN | WWW | SSS >\n", argv[0]);
        
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const int device_id = atoi(argv[1]);
    const int N = atoi(argv[2]);
    const char *library = argv[3];
    const int q = (int)(sqrt((double)nranks) + 0.5);

    const int use_nvshmem = (strcmp(library, "WWW") == 0 || strcmp(library, "SSS") == 0);

    if (use_nvshmem)
    {
        nvshmemx_init_attr_t attr;
        memset(&attr, 0, sizeof(attr));
        MPI_Comm mpi_comm = MPI_COMM_WORLD;
        attr.mpi_comm = &mpi_comm;
        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    }

    cudaSetDevice(device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    /*
     * Domain Decomposition - 2D Block Matrix
     */

    const int proc_row = rank / q;
    const int proc_col = rank % q;
    const int bs = N / q;
    const size_t block_elems_sz = (size_t)bs * bs;
    const int block_elems = (int)block_elems_sz;
    const size_t block_bytes = block_elems_sz * sizeof(double);

    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, proc_row, proc_col, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, proc_col, proc_row, &col_comm);

    printf("mmb node=%s rank=%d grid=(%d,%d) device=%d:%s library=%s SUMMA\n", name, rank, proc_row, proc_col, device_id, prop.name, library);

    double *h_Aij = (double *)malloc(block_bytes);
    double *h_Bij = (double *)malloc(block_bytes);
    double *h_Cij = (double *)malloc(block_bytes);
   
    double *A = NULL, *B = NULL, *C = NULL, *tmp = NULL;
   
    if (rank == 0) 
    {
        const size_t matrix_elems = (size_t)N * N;
        A = (double *)malloc(matrix_elems * sizeof(double));
        B = (double *)malloc(matrix_elems * sizeof(double));
        C = (double *)malloc(matrix_elems * sizeof(double));
        tmp = (double *)malloc(block_bytes);
    
        for (size_t i = 0; i < matrix_elems; ++i) 
        {
            A[i] = 1.0;
            B[i] = 2.0;
            C[i] = 0.0;
        }
    }

    if (rank == 0) 
    {
        for (int r = 0; r < nranks; ++r) 
        {
            const int br = r / q, bc = r % q;
           
            pack_block(A, tmp, N, bs, br, bc);
           
            if (r == 0) 
                memcpy(h_Aij, tmp, block_bytes);
            else 
                MPI_Send(tmp, block_elems, MPI_DOUBLE, r, 100, MPI_COMM_WORLD);

            pack_block(B, tmp, N, bs, br, bc);

            if (r == 0) 
                memcpy(h_Bij, tmp, block_bytes);
            else 
                MPI_Send(tmp, block_elems, MPI_DOUBLE, r, 200, MPI_COMM_WORLD);
        }
    } else 
    {
        MPI_Recv(h_Aij, block_elems, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(h_Bij, block_elems, MPI_DOUBLE, 0, 200, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    double *d_Cij = NULL;
    cudaMalloc((void **)&d_Cij, block_bytes);
    cudaMemset(d_Cij, 0, block_bytes);

    double *d_Aij_local = NULL, *d_Bij_local = NULL;
   
    if (strcmp(library, "CCC") == 0 || strcmp(library, "XXX") == 0 || strcmp(library, "NNN") == 0 || strcmp(library, "WWW") == 0 || strcmp(library, "SSS") == 0) 
    {
        if (strcmp(library, "WWW") == 0 || strcmp(library, "SSS") == 0)
        {
            d_Aij_local = (double *)nvshmem_malloc(block_bytes);
            d_Bij_local = (double *)nvshmem_malloc(block_bytes);
        }
        else
        {
            cudaMalloc((void **)&d_Aij_local, block_bytes);
            cudaMalloc((void **)&d_Bij_local, block_bytes);
        }

        cudaMemcpy(d_Aij_local, h_Aij, block_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Bij_local, h_Bij, block_bytes, cudaMemcpyHostToDevice);
    }

    ncclComm_t nccl_row, nccl_col;

    if (strcmp(library, "NNN") == 0) 
    {
        ncclUniqueId row_id, col_id;

        if (proc_col == 0) 
            ncclGetUniqueId(&row_id);
        if (proc_row == 0) 
            ncclGetUniqueId(&col_id);

        MPI_Bcast(&row_id, sizeof(row_id), MPI_BYTE, 0, row_comm);
        MPI_Bcast(&col_id, sizeof(col_id), MPI_BYTE, 0, col_comm);

        ncclGroupStart();
        ncclCommInitRank(&nccl_row, q, row_id, proc_col);
        ncclCommInitRank(&nccl_col, q, col_id, proc_row);
        ncclGroupEnd();
    }

    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);

  /**********************************************/
  /**/ const double start_time = MPI_Wtime(); /**/
  /**********************************************/

    if (strcmp(library, "MMM") == 0) 
    {
        /*
         * MMM = MPI-SYNC
         * 
         */

        double *h_Apanel = (double *)malloc(block_bytes);
        double *h_Bpanel = (double *)malloc(block_bytes);
        double *d_Apanel = NULL, *d_Bpanel = NULL;
        
        cudaMalloc((void **)&d_Apanel, block_bytes);
        cudaMalloc((void **)&d_Bpanel, block_bytes);

        for (int k = 0; k < q; ++k) 
        {
            if (proc_col == k) 
                memcpy(h_Apanel, h_Aij, block_bytes);
            
            MPI_Bcast(h_Apanel, block_elems, MPI_DOUBLE, k, row_comm);

            if (proc_row == k) 
                memcpy(h_Bpanel, h_Bij, block_bytes);
            
            MPI_Bcast(h_Bpanel, block_elems, MPI_DOUBLE, k, col_comm);

            cudaMemcpy(d_Apanel, h_Apanel, block_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_Bpanel, h_Bpanel, block_bytes, cudaMemcpyHostToDevice);

            ABMultiplyAccumulate(d_Apanel, d_Bpanel, d_Cij, bs, bs, bs, bs, bs, bs, TILE_DIM);
            
            cudaGetLastError();
            cudaDeviceSynchronize();
        }

        cudaFree(d_Apanel);
        cudaFree(d_Bpanel);
        free(h_Apanel);
        free(h_Bpanel);

    } else if (strcmp(library, "CCC") == 0) 
    {
        /*
         * CCC = CUDA-Aware MPI-SYNC
         * 
         */
        double *d_Apanel = NULL, *d_Bpanel = NULL;
        cudaMalloc((void **)&d_Apanel, block_bytes);
        cudaMalloc((void **)&d_Bpanel, block_bytes);

        for (int k = 0; k < q; ++k)
        {
            double *A_buf = (proc_col == k) ? d_Aij_local : d_Apanel;
            double *B_buf = (proc_row == k) ? d_Bij_local : d_Bpanel;

            MPI_Bcast(A_buf, block_elems, MPI_DOUBLE, k, row_comm);
            MPI_Bcast(B_buf, block_elems, MPI_DOUBLE, k, col_comm);

            ABMultiplyAccumulate(A_buf, B_buf, d_Cij,
                                 bs, bs, bs, bs, bs, bs, TILE_DIM);
            cudaGetLastError();
            cudaDeviceSynchronize();
        }

        cudaFree(d_Apanel);
        cudaFree(d_Bpanel);

    } else if (strcmp(library, "YYY") == 0)
    {
        /*
         * YYY-CHUNKED-ASYNC
         */
        
        int chunk_k = 1024;
        const char *chunk_env = getenv("YYY_CHUNK_K");

        if (chunk_env && atoi(chunk_env) > 0)
            chunk_k = atoi(chunk_env);
        
        if (chunk_k > bs) chunk_k = bs;

        const int max_chunk = chunk_k;
        const size_t a_chunk_bytes = (size_t)bs * max_chunk * sizeof(double);
        const size_t b_chunk_bytes = (size_t)max_chunk * bs * sizeof(double);

        double *h_Achunk[2] = {NULL, NULL};
        double *h_Bchunk[2] = {NULL, NULL};
        double *d_Achunk[2] = {NULL, NULL};
        double *d_Bchunk[2] = {NULL, NULL};

        MPI_Request reqA[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        MPI_Request reqB[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        cudaStream_t transfer_stream, compute_stream;
        cudaEvent_t h2d_ready[2], h2d_done[2], compute_done[2];

        for (int s = 0; s < 2; ++s) 
        {
            cudaHostAlloc((void **)&h_Achunk[s], a_chunk_bytes, cudaHostAllocDefault);
            cudaHostAlloc((void **)&h_Bchunk[s], b_chunk_bytes, cudaHostAllocDefault);
            cudaMalloc((void **)&d_Achunk[s], a_chunk_bytes);
            cudaMalloc((void **)&d_Bchunk[s], b_chunk_bytes);
            cudaEventCreateWithFlags(&h2d_ready[s], cudaEventDisableTiming);
            cudaEventCreateWithFlags(&h2d_done[s], cudaEventDisableTiming);
            cudaEventCreateWithFlags(&compute_done[s], cudaEventDisableTiming);
           
            cudaEventRecord(h2d_done[s], 0);
            cudaEventRecord(compute_done[s], 0);
        }
        cudaStreamCreateWithFlags(&transfer_stream, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);

        const int chunks_per_panel = (bs + chunk_k - 1) / chunk_k;
        const int total_tasks = q * chunks_per_panel;

        int task_k[2] = {0, 0};
        int task_k0[2] = {0, 0};
        int task_kc[2] = {0, 0};

        task_k[0] = 0;
        task_k0[0] = 0;
        task_kc[0] = (chunk_k < bs) ? chunk_k : bs;
        
        post_summa_chunk(task_k[0], task_k0[0], task_kc[0], 0, proc_row, proc_col, bs, h_Aij, h_Bij, h_Achunk, h_Bchunk, row_comm, col_comm, reqA, reqB);

        for (int t = 0; t < total_tasks; ++t) 
        {
            const int cur = t & 1;
            const int next = cur ^ 1;
            const int kc = task_kc[cur];

            MPI_Wait(&reqA[cur], MPI_STATUS_IGNORE);
            MPI_Wait(&reqB[cur], MPI_STATUS_IGNORE);

            cudaEventSynchronize(compute_done[cur]);

            const size_t a_bytes = (size_t)bs * kc * sizeof(double);
            const size_t b_bytes = (size_t)kc * bs * sizeof(double);

            cudaMemcpyAsync(d_Achunk[cur], h_Achunk[cur], a_bytes, cudaMemcpyHostToDevice, transfer_stream);
            cudaMemcpyAsync(d_Bchunk[cur], h_Bchunk[cur], b_bytes, cudaMemcpyHostToDevice, transfer_stream);
            cudaEventRecord(h2d_ready[cur], transfer_stream);
            cudaEventRecord(h2d_done[cur], transfer_stream);

            cudaStreamWaitEvent(compute_stream, h2d_ready[cur], 0);
            ABMultiplyAccumulateAsync(d_Achunk[cur], d_Bchunk[cur], d_Cij, bs, bs, kc, 0, kc, kc, bs, bs, TILE_DIM, compute_stream);
            
            cudaGetLastError();
            cudaEventRecord(compute_done[cur], compute_stream);
            
            if (t + 1 < total_tasks) 
            {
                const int nt = t + 1;
                const int nk = nt / chunks_per_panel;
                const int nc = nt % chunks_per_panel;
                const int nk0 = nc * chunk_k;
                const int nkc = ((nk0 + chunk_k) <= bs) ? chunk_k : (bs - nk0);

                cudaEventSynchronize(h2d_done[next]);

                task_k[next] = nk;
                task_k0[next] = nk0;
                task_kc[next] = nkc;

                post_summa_chunk(nk, nk0, nkc, next, proc_row, proc_col, bs, h_Aij, h_Bij, h_Achunk, h_Bchunk, row_comm, col_comm, reqA, reqB);
            }
        }

        cudaStreamSynchronize(compute_stream);
        cudaStreamSynchronize(transfer_stream);

        cudaStreamDestroy(transfer_stream);
        cudaStreamDestroy(compute_stream);

        for (int s = 0; s < 2; ++s) 
        {
            cudaEventDestroy(h2d_ready[s]);
            cudaEventDestroy(h2d_done[s]);
            cudaEventDestroy(compute_done[s]);
            cudaFree(d_Achunk[s]);
            cudaFree(d_Bchunk[s]);
            cudaFreeHost(h_Achunk[s]);
            cudaFreeHost(h_Bchunk[s]);
        }
    } else if (strcmp(library, "XXX") == 0)
    {
        /*
         * XXX = CUDA-Aware MPI-ASYNC
         */

        int chunk_k = 1024;
        const char *chunk_env = getenv("XXX_CHUNK_K");
        if (chunk_env && atoi(chunk_env) > 0)
            chunk_k = atoi(chunk_env);
        if (chunk_k > bs) chunk_k = bs;

        const int chunks_per_panel = (bs + chunk_k - 1) / chunk_k;
        const size_t max_b_chunk_bytes = (size_t)chunk_k * bs * sizeof(double);

        double *d_Apanel = NULL;
        double *d_Bchunk[2] = {NULL, NULL};

        cudaMalloc((void **)&d_Apanel, block_bytes);
        cudaMalloc((void **)&d_Bchunk[0], max_b_chunk_bytes);
        cudaMalloc((void **)&d_Bchunk[1], max_b_chunk_bytes);

        cudaStream_t compute_stream;
        cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);
        cudaEvent_t compute_done[2];

        for (int s = 0; s < 2; ++s)
        {
            cudaEventCreateWithFlags(&compute_done[s], cudaEventDisableTiming);
            cudaEventRecord(compute_done[s], compute_stream);
        }

        for (int k = 0; k < q; ++k)
        {
            double *A_buf = (proc_col == k) ? d_Aij_local : d_Apanel;
            MPI_Request reqA = MPI_REQUEST_NULL;
            MPI_Ibcast(A_buf, block_elems, MPI_DOUBLE, k, row_comm, &reqA);

            MPI_Request reqB[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
            int kc_slot[2] = {0, 0};
            int k0_slot[2] = {0, 0};

            int k0 = 0;
            int kc = (chunk_k < bs) ? chunk_k : bs;
            double *B0 = (proc_row == k) ? (d_Bij_local + (size_t)k0 * bs) : d_Bchunk[0];
            kc_slot[0] = kc;
            k0_slot[0] = k0;
            
            MPI_Ibcast(B0, kc * bs, MPI_DOUBLE, k, col_comm, &reqB[0]);

            MPI_Wait(&reqA, MPI_STATUS_IGNORE);

            for (int c = 0; c < chunks_per_panel; ++c)
            {
                const int cur = c & 1;
                const int next = cur ^ 1;

                MPI_Wait(&reqB[cur], MPI_STATUS_IGNORE);

                const int cur_k0 = k0_slot[cur];
                const int cur_kc = kc_slot[cur];
                const double *B_cur = (proc_row == k) ? (d_Bij_local + (size_t)cur_k0 * bs) : d_Bchunk[cur];

                if (c + 1 < chunks_per_panel)
                {
                    const int next_k0 = (c + 1) * chunk_k;
                    const int next_kc = ((next_k0 + chunk_k) <= bs) ? chunk_k : (bs - next_k0);

                    if (proc_row != k)
                        cudaEventSynchronize(compute_done[next]);

                    double *B_next = (proc_row == k) ? (d_Bij_local + (size_t)next_k0 * bs) : d_Bchunk[next];

                    k0_slot[next] = next_k0;
                    kc_slot[next] = next_kc;
                    MPI_Ibcast(B_next, next_kc * bs, MPI_DOUBLE, k, col_comm, &reqB[next]);
                }

                ABMultiplyAccumulateAsync(A_buf, B_cur, d_Cij, bs, bs, bs, cur_k0, cur_kc, bs, bs, bs, TILE_DIM, compute_stream);
                
                cudaGetLastError();
                cudaEventRecord(compute_done[cur], compute_stream);
            }

            cudaStreamSynchronize(compute_stream);
        }

        for (int s = 0; s < 2; ++s)
        {
            cudaEventDestroy(compute_done[s]);
            cudaFree(d_Bchunk[s]);
        }
        cudaStreamDestroy(compute_stream);
        cudaFree(d_Apanel);

    } else if (strcmp(library, "SSS") == 0)
    {
        /*
         * SSS = NVSHMEM-CHUNKED-ASYNC
         */

        int chunk_k = 1024;
        const char *env = getenv("SSS_CHUNK_K");
        if (env && atoi(env) > 0)
            chunk_k = atoi(env);
        if (chunk_k > bs)
            chunk_k = bs;

        const int nchunks = (bs + chunk_k - 1) / chunk_k;
        const size_t max_b_chunk_bytes = (size_t)chunk_k * bs * sizeof(double);

        double *d_Apanel = NULL;
        double *d_Bchunk[2] = {NULL, NULL};
        cudaStream_t comm_stream, compute_stream;
        cudaEvent_t a_ready, ready[2], done[2];

        cudaMalloc((void **)&d_Apanel, block_bytes);
        cudaMalloc((void **)&d_Bchunk[0], max_b_chunk_bytes);
        cudaMalloc((void **)&d_Bchunk[1], max_b_chunk_bytes);
        cudaStreamCreateWithFlags(&comm_stream, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);
        cudaEventCreateWithFlags(&a_ready, cudaEventDisableTiming);

        for (int s = 0; s < 2; ++s)
        {
            cudaEventCreateWithFlags(&ready[s], cudaEventDisableTiming);
            cudaEventCreateWithFlags(&done[s], cudaEventDisableTiming);
            cudaEventRecord(done[s], compute_stream);
        }

        nvshmem_barrier_all();

        for (int k = 0; k < q; ++k)
        {
            const int a_pe = proc_row * q + k;
            const int b_pe = k * q + proc_col;
            const double *A_cur = NULL;

            if (a_pe == rank)
            {
                A_cur = d_Aij_local;
                cudaEventRecord(a_ready, comm_stream);
            }
            else
            {
                nvshmemx_getmem_nbi_on_stream(d_Apanel, d_Aij_local, block_bytes, a_pe, comm_stream);
                nvshmemx_quiet_on_stream(comm_stream);
                cudaEventRecord(a_ready, comm_stream);
                A_cur = d_Apanel;
            }

            cudaStreamWaitEvent(compute_stream, a_ready, 0);

            for (int c = 0; c < nchunks; ++c)
            {
                const int slot = c & 1;
                const int k0 = c * chunk_k;
                const int kc = (chunk_k < (bs - k0)) ? chunk_k : (bs - k0);
                const size_t chunk_bytes = (size_t)kc * bs * sizeof(double);
                const double *B_cur = NULL;

                cudaStreamWaitEvent(comm_stream, done[slot], 0);

                if (b_pe == rank)
                {
                    B_cur = d_Bij_local + (size_t)k0 * bs;
                    cudaEventRecord(ready[slot], comm_stream);
                }
                else
                {
                    const double *remote_src = d_Bij_local + (size_t)k0 * bs;
                    nvshmemx_getmem_nbi_on_stream(d_Bchunk[slot], remote_src, chunk_bytes, b_pe, comm_stream);
                    nvshmemx_quiet_on_stream(comm_stream);
                    cudaEventRecord(ready[slot], comm_stream); 
                    B_cur = d_Bchunk[slot];
                }

                cudaStreamWaitEvent(compute_stream, ready[slot], 0);
                
                ABMultiplyAccumulateAsync(A_cur, B_cur, d_Cij, bs, bs, bs, k0, kc, bs, bs, bs, TILE_DIM, compute_stream);
               
                cudaGetLastError();
                cudaEventRecord(done[slot], compute_stream);
            }

            cudaStreamSynchronize(compute_stream);
        }

        cudaStreamSynchronize(comm_stream);
        cudaStreamSynchronize(compute_stream);
        nvshmem_barrier_all();

        cudaEventDestroy(a_ready);
        for (int s = 0; s < 2; ++s)
        {
            cudaEventDestroy(ready[s]);
            cudaEventDestroy(done[s]);
            cudaFree(d_Bchunk[s]);
        }
        cudaStreamDestroy(comm_stream);
        cudaStreamDestroy(compute_stream);
        cudaFree(d_Apanel);

    } else if (strcmp(library, "WWW") == 0)
    {
        /*
         * WWW = NVSHMEM-SYNC
         */
        double *d_Apanel = (double *)nvshmem_malloc(block_bytes);
        double *d_Bpanel = (double *)nvshmem_malloc(block_bytes);

        nvshmem_barrier_all();

        for (int k = 0; k < q; ++k)
        {
            const int a_pe = proc_row * q + k;
            const int b_pe = k * q + proc_col;

            nvshmem_getmem(d_Apanel, d_Aij_local, block_bytes, a_pe);
            nvshmem_getmem(d_Bpanel, d_Bij_local, block_bytes, b_pe);

            ABMultiplyAccumulate(d_Apanel, d_Bpanel, d_Cij, bs, bs, bs, bs, bs, bs, TILE_DIM);
            cudaDeviceSynchronize();

            nvshmem_barrier_all();
        }

        nvshmem_free(d_Apanel);
        nvshmem_free(d_Bpanel);

    } else if (strcmp(library, "NNN") == 0) 
    {
        /*
         * NNN = NCCL
         */

        int chunk_k = 1024;
        const char *env = getenv("NNN_CHUNK_K");

        if (env && atoi(env) > 0) 
            chunk_k = atoi(env);

        if (chunk_k > bs) 
            chunk_k = bs;

        const int nchunks = (bs + chunk_k - 1) / chunk_k;
        double *a_panel = NULL, *b_chunk[2] = {NULL, NULL};
        cudaStream_t comm_stream, compute_stream;
        cudaEvent_t ready[2], done[2];
        cudaMalloc((void **)&a_panel, block_bytes);
        cudaStreamCreateWithFlags(&comm_stream, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);

        for (int i = 0; i < 2; ++i) 
        {
            cudaMalloc((void **)&b_chunk[i], (size_t)chunk_k * bs * sizeof(double));
            cudaEventCreateWithFlags(&ready[i], cudaEventDisableTiming);
            cudaEventCreateWithFlags(&done[i], cudaEventDisableTiming);
            cudaEventRecord(done[i], compute_stream);
        }

        for (int k = 0; k < q; ++k) 
        {
            const double *a_send = proc_col == k ? d_Aij_local : a_panel;
           
            for (int c = 0; c < nchunks; ++c) 
            {
                const int slot = c & 1, k0 = c * chunk_k;
                const int kc = chunk_k < bs - k0 ? chunk_k : bs - k0;

                cudaStreamWaitEvent(comm_stream, done[slot], 0);
                const double *b_send = proc_row == k ? d_Bij_local + (size_t)k0 * bs : b_chunk[slot];
               
                ncclGroupStart();

                if (c == 0)
                    ncclBroadcast(a_send, a_panel, block_elems_sz, ncclDouble, k, nccl_row, comm_stream);
                
                ncclBroadcast(b_send, b_chunk[slot], (size_t)kc * bs, ncclDouble, k, nccl_col, comm_stream);
                ncclGroupEnd();

                cudaEventRecord(ready[slot], comm_stream);
                cudaStreamWaitEvent(compute_stream, ready[slot], 0);

                ABMultiplyAccumulateAsync(a_panel, b_chunk[slot], d_Cij, bs, bs, bs, k0, kc, bs, bs, bs, TILE_DIM, compute_stream);
                
                cudaGetLastError();
                cudaEventRecord(done[slot], compute_stream);
            }
            
            cudaStreamSynchronize(compute_stream);
        }

        cudaStreamSynchronize(comm_stream);
         
        for (int i = 0; i < 2; ++i) 
        {
            cudaEventDestroy(ready[i]);
            cudaEventDestroy(done[i]);
            cudaFree(b_chunk[i]);
        }
        cudaFree(a_panel);
        cudaStreamDestroy(comm_stream);
        cudaStreamDestroy(compute_stream);
    }

    cudaMemcpy(h_Cij, d_Cij, block_bytes, cudaMemcpyDeviceToHost);

    if (rank == 0) 
    {
        unpack_block(C, h_Cij, N, bs, 0, 0);

        for (int r = 1; r < nranks; ++r) 
        {
            MPI_Recv(tmp, block_elems, MPI_DOUBLE, r, 300, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            unpack_block(C, tmp, N, bs, r / q, r % q);
        }

    } else 
    {
        MPI_Send(h_Cij, block_elems, MPI_DOUBLE, 0, 300, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

  /***********************************************/
  /**/  const double stop_time = MPI_Wtime();  /**/
  /***********************************************/

    if (rank == 0) 
    {
        printf("Matrix size = %d | Library = %s | Decomposition = SUMMA %dx%d\n", N, library, q, q);
        printf("Time = %.6f s\n", stop_time - start_time);

        double *d_C = NULL;
        cudaMalloc((void **)&d_C, (size_t)N * N * sizeof(double));
        cudaMemcpy(d_C, C, (size_t)N * N * sizeof(double), cudaMemcpyHostToDevice);
        
        /*
         * Validate the matrix values
         */

        validate_matrix_C(d_C, N, N, 2.0 * N, 1.0e-9, 1.0e-12, rank);
        
        cudaFree(d_C);
    }


    /* ------------------------------------------------------------------ */
    /* Cleanup: free only buffers that were actually allocated.           */
    /* ------------------------------------------------------------------ */

    if (d_Aij_local)
    {
        if (use_nvshmem) nvshmem_free(d_Aij_local);
        else cudaFree(d_Aij_local);
    }
    
    if (d_Bij_local)
    {
        if (use_nvshmem) nvshmem_free(d_Bij_local);
        else cudaFree(d_Bij_local);
    }
    
    cudaFree(d_Cij);
    free(h_Aij);
    free(h_Bij);
    free(h_Cij);

    if (rank == 0) 
    {
        free(A); 
        free(B); 
        free(C); 
        free(tmp);
    }

    if (strcmp(library, "NNN") == 0) 
    {
        ncclCommDestroy(nccl_row);
        ncclCommDestroy(nccl_col);
    }
    
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);

    if (use_nvshmem)
        nvshmem_finalize();

    MPI_Finalize();

    return EXIT_SUCCESS;
}

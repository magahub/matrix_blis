/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"
#include "blix.h"

typedef struct
{
    obj_t* a;
    obj_t* b;
    obj_t* c;
    cntx_t* cntx;
    cntl_t* cntl;
    thrinfo_t* thread;
} gemm_params;

void blx_gemm_blk_var2_thread( tci_comm* comm,
                               uint64_t gid,
                               uint64_t unused,
                               void* param_ )
{
    gemm_params* param = param_;

    obj_t b1, c1;
    dim_t i;
    dim_t b_alg;
    dim_t my_start, my_end;

    // Determine the current thread's subpartition range.
    bli_thread_get_range_ndim
    (
      BLIS_FWD, comm->ngang, gid, param->a, param->b, param->c, param->cntl,
      param->cntx, &my_start, &my_end
    );

    // Partition along the m dimension.
    for ( i = my_start; i < my_end; i += b_alg )
    {
        // Determine the current algorithmic blocksize.
        b_alg = blx_determine_blocksize_f( i, my_end, param->c,
                                           bli_cntl_bszid( param->cntl ),
                                           param->cntx );

        // Acquire partitions for B1 and C1.
        bli_acquire_mpart_ndim( BLIS_FWD, BLIS_SUBPART1, i, b_alg, param->b, &b1 );
        bli_acquire_mpart_ndim( BLIS_FWD, BLIS_SUBPART1, i, b_alg, param->c, &c1 );

        // Perform gemm subproblem.
        blx_gemm_int
        (
          param->a, &b1, &c1, param->cntx,
          bli_cntl_sub_node( param->cntl ),
          bli_thrinfo_sub_node( param->thread )
        );
    }
}

void blx_gemm_blk_var2
     (
       obj_t*  a,
       obj_t*  b,
       obj_t*  c,
       cntx_t* cntx,
       cntl_t* cntl,
       thrinfo_t* thread
     )
{
    gemm_params param;

    param.a = a;
    param.b = b;
    param.c = c;
    param.cntx = cntx;
    param.cntl = cntl;
    param.thread = thread;

    tci_comm* comm = thread->comm;
    tci_range range = {comm->ngang, 1};

    tci_comm_distribute_over_gangs( comm,
                                    range,
                                    blx_gemm_blk_var2_thread,
                                    &param );
}


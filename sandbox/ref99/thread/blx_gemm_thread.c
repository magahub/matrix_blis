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
    gemmint_t func;
    opid_t family;
    obj_t* a;
    obj_t* b;
    obj_t* c;
    cntx_t* cntx;
    rntm_t* rntm;
    cntl_t* cntl;
} l3_thrinfo_t;

void blx_gemm_thread_int( tci_comm* comm, void* thrinfo_ )
{
    l3_thrinfo_t* thrinfo = (l3_thrinfo_t*)thrinfo_;

    dim_t      id = comm->tid;

    cntl_t*    cntl_use;
    thrinfo_t* thread;

    // Create a default control tree for the operation, if needed.
    blx_l3_cntl_create_if( thrinfo->family, thrinfo->a, thrinfo->b,
                           thrinfo->c, thrinfo->cntl, &cntl_use );

    // Create the root node of the current thread's thrinfo_t structure.
    thrinfo_t* glb_thread = bli_thrinfo_create( comm, FALSE, NULL );
    thrinfo_t* thread = bli_thrinfo_create_for_cntl( thrinfo->rntm,
                                                     thrinfo->cntl,
                                                     glb_thread );

    thrinfo->func
    (
      thrinfo->a,
      thrinfo->b,
      thrinfo->c,
      thrinfo->cntx,
      thrinfo->rntm,
      cntl_use,
      thread
    );

    // Free the control tree, if one was created locally.
    blx_l3_cntl_free_if( thrinfo->a, thrinfo->b, thrinfo->c, thrinfo->cntl,
                         cntl_use, thread );

    // Free the current thread's thrinfo_t structure.
    bli_thrinfo_free( thread );
    bli_free_intl( glb_thread );
}
void blx_gemm_thread
     (
       gemmint_t func,
       opid_t    family,
       obj_t*    a,
       obj_t*    b,
       obj_t*    c,
       cntx_t*   cntx,
       rntm_t*   rntm,
       cntl_t*   cntl
     )
{
    l3_thrinfo_t info;

    info.func = func;
    info.family = family;
    info.a = a;
    info.b = b;
    info.c = c;
    info.cntx = cntx;
    info.rntm = rntm;
    info.cntl = cntl;

    dim_t n_threads = bli_rntm_num_threads( rntm );

    tci_parallelize( blx_gemm_thread_int, &info, n_threads, 0 );
}

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

#ifndef BLIS_THRINFO_H
#define BLIS_THRINFO_H

// Include thread communicator object definitions and prototypes.
#include "tci.h"

// Thread info structure definition
struct thrinfo_s
{
	// The thread communicator for the other threads sharing the same work
	// at this level.
	tci_comm*          comm;

    // When freeing, should the communicators in this node be freed? Usually,
    // this is field is true, but when nodes are created that share the same
    // communicators as other nodes (such as with packm nodes), this is set
    // to false.
    bool_t             free_comm;

	struct thrinfo_s*  sub_node;
};
typedef struct thrinfo_s thrinfo_t;

//
// thrinfo_t functions
// NOTE: The naming of these should be made consistent at some point.
// (ie: bli_thrinfo_ vs. bli_thread_)
//

// thrinfo_t query (field only)

static dim_t bli_thread_num_threads( thrinfo_t* t )
{
	return t->comm->nthread;
}

static dim_t bli_thread_ocomm_id( thrinfo_t* t )
{
	return t->comm->tid;
}

static dim_t bli_thread_n_way( thrinfo_t* t )
{
	return t->comm->ngang;
}

static dim_t bli_thread_work_id( thrinfo_t* t )
{
	return t->comm->gid;
}

static tci_comm* bli_thrinfo_ocomm( thrinfo_t* t )
{
	return t->comm;
}

static bool_t bli_thrinfo_needs_free_comm( thrinfo_t* t )
{
	return t->free_comm;
}

static thrinfo_t* bli_thrinfo_sub_node( thrinfo_t* t )
{
	return t->sub_node;
}

// thrinfo_t query (complex)

static bool_t bli_thread_am_ochief( thrinfo_t* t )
{
	return t->comm->tid == 0;
}

// thrinfo_t modification

static void bli_thrinfo_set_sub_node( thrinfo_t* sub_node, thrinfo_t* t )
{
	t->sub_node = sub_node;
}

// other thrinfo_t-related functions

static void* bli_thread_obroadcast( thrinfo_t* t, void* p )
{
    tci_comm_bcast( bli_thrinfo_ocomm( t ), &p, 0 );
    return p;
}

static void bli_thread_obarrier( thrinfo_t* t )
{
    tci_comm_barrier( bli_thrinfo_ocomm( t ) );
}


//
// Prototypes for level-3 thrinfo functions not specific to any operation.
//

thrinfo_t* bli_thrinfo_create
     (
       tci_comm*  comm,
       bool_t     free_comm,
       thrinfo_t* sub_node
     );

void bli_thrinfo_free
     (
       thrinfo_t* thread
     );

void bli_thrinfo_init
     (
       thrinfo_t* thread,
       tci_comm*  comm,
       bool_t     free_comm,
       thrinfo_t* sub_node
     );

void bli_thrinfo_init_single
     (
       thrinfo_t* thread
     );

// -----------------------------------------------------------------------------

thrinfo_t* bli_thrinfo_create_for_cntl
     (
       cntx_t*    cntx,
       cntl_t*    cntl_chl,
       thrinfo_t* thread_par
     );

void bli_thrinfo_grow
     (
       cntx_t*    cntx,
       cntl_t*    cntl,
       thrinfo_t* thread
     );

#endif

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

thrinfo_t* bli_thrinfo_create
     (
       tci_comm*  comm,
       bool_t     free_comm,
       thrinfo_t* sub_node
     )
{
    thrinfo_t* thread = bli_malloc_intl( sizeof( thrinfo_t ) );

    bli_thrinfo_init
	(
	  thread,
	  comm,
	  free_comm,
	  sub_node
	);

    return thread;
}

void bli_thrinfo_init
     (
       thrinfo_t* thread,
       tci_comm*  comm,
       bool_t     free_comm,
       thrinfo_t* sub_node
     )
{
	thread->comm      = comm;
	thread->free_comm = free_comm;
	thread->sub_node  = sub_node;
}

void bli_thrinfo_init_single
     (
       thrinfo_t* thread
     )
{
	bli_thrinfo_init
	(
	  thread,
	  tci_single,
	  FALSE,
	  thread
	);
}

// -----------------------------------------------------------------------------

#include "assert.h"

thrinfo_t* bli_thrinfo_create_for_cntl
     (
       cntx_t*    cntx,
       cntl_t*    cntl_par,
       cntl_t*    cntl_chl,
       thrinfo_t* thread_par
     )
{
	bszid_t bszid_chl = bli_cntl_bszid( cntl_chl );
	dim_t child_n_way = bli_cntx_way_for_bszid( bszid_chl, cntx );

	tci_comm* new_comm = bli_malloc_intl( sizeof( tci_comm ) );
	tci_gang( bli_thrinfo_ocomm( thread_par ), &new_comm, TCI_EVENLY, child_n_way, 0 );

	// All threads create a new thrinfo_t node using the communicator
	// that was created by their chief, as identified by parent_work_id.
	thrinfo_t* thread_chl = bli_thrinfo_create
	(
	  new_comm,
	  TRUE,
	  NULL
	);

	return thread_chl;
}

void bli_thrinfo_grow
     (
       cntx_t*    cntx,
       cntl_t*    cntl,
       thrinfo_t* thread
     )
{
	// If the sub-node of the thrinfo_t object is non-NULL, we don't
	// need to create it, and will just use the existing sub-node as-is.
	if ( bli_thrinfo_sub_node( thread ) != NULL ) return;

	// Create a new node (or, if needed, multiple nodes) and return the
	// pointer to the (eldest) child.
	cntl_t* cntl_child = bli_cntl_sub_node( cntl );
	thrinfo_t* thread_child;

	// We must handle two cases: those where the next node in the
    // control tree is a partitioning node, and those where it is
    // a non-partitioning (ie: packing) node.
    if ( bli_cntl_bszid( cntl_child ) != BLIS_NO_PART )
    {
        // Create the child thrinfo_t node corresponding to cntl_cur,
        // with cntl_par being the parent.
        thread_child = bli_thrinfo_create_for_cntl
        (
          cntx,
          cntl,
          cntl_child,
          thread
        );
    }
    else // if ( bli_cntl_bszid( cntl_cur ) == BLIS_NO_PART )
    {
        // Create a thrinfo_t node corresponding to cntl_cur. Notice that
        // the free_comm field is set to FALSE, since cntl_cur is a
        // non-partitioning node. The communicator used here will be
        // freed when thread_seg, or one of its descendents, is freed.
        thread_child = bli_thrinfo_create
        (
          bli_thrinfo_ocomm( thread ),
          FALSE,
          NULL
        );
    }

	// Attach the child thrinfo_t node to its parent structure.
	bli_thrinfo_set_sub_node( thread_child, thread );
}

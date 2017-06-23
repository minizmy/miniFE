#ifndef _exchange_externals_hpp_
#define _exchange_externals_hpp_

//@HEADER
// ************************************************************************
// 
//               HPCCG: Simple Conjugate Gradient Benchmark Code
//                 Copyright (2006) Sandia Corporation
// 
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
// 
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//  
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//  
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
// Questions? Contact Michael A. Heroux (maherou@sandia.gov) 
// 
// ************************************************************************
//@HEADER

#include <cstdlib>
#include <iostream>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include <outstream.hpp>

#include <TypeTraits.hpp>

namespace miniFE {

#ifdef HAVE_MPI
struct recv_args {
  std::vector <MINIFE_SCALAR> & x_coefs;
  std::vector<MPI_Request>& request;
  const std::vector<MINIFE_LOCAL_ORDINAL>& recv_length;
  MPI_Datatype mpi_dtype;
  const std::vector<int>& neighbors;
  int local_nrow;

};

struct send_args {
  std::vector<MINIFE_SCALAR>& send_buffer;
  const std::vector<MINIFE_LOCAL_ORDINAL>& send_length;
  MPI_Datatype mpi_dtype;
  const std::vector<int>& neighbors;
};

void
post_recvs(const size_t i, const size_t stop, void *arg)
{
  struct recv_args* a = (recv_args*) arg;
  MINIFE_SCALAR* x_external = &(a->x_coefs[a->local_nrow]);

  // Post receives first
  int n_recv = a->recv_length[i];
  MPI_Irecv(x_external, n_recv, a->mpi_dtype, a->neighbors[i], 99,
            MPI_COMM_WORLD, &a->request[i]);
  x_external += n_recv;
}

void
send_func(const size_t i, const size_t stop, void *arg)
{
  struct send_args *a = (send_args*)arg;
  MINIFE_SCALAR* s_buffer = &a->send_buffer[0];

	for(int j = 0; j < i; ++j) {
		s_buffer += a->send_length[j];
	}

	int n_send = a->send_length[i];

  MPI_Send(s_buffer, n_send, a->mpi_dtype, a->neighbors[i], 99, MPI_COMM_WORLD);
}

#endif

template<typename MatrixType,
         typename VectorType>
void
exchange_externals(MatrixType& A,
                   VectorType& x)
{
#ifdef HAVE_MPI
#ifdef MINIFE_DEBUG
  std::ostream& os = outstream();
  os << "entering exchange_externals\n";
#endif

  int numprocs = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

  if (numprocs < 2) return;

  typedef typename MatrixType::ScalarType Scalar;
  typedef typename MatrixType::LocalOrdinalType LocalOrdinal;
  typedef typename MatrixType::GlobalOrdinalType GlobalOrdinal;

  // Extract Matrix pieces

  int local_nrow = A.rows.size();
  int num_neighbors = A.neighbors.size();
  const std::vector<LocalOrdinal>& recv_length = A.recv_length;
  const std::vector<LocalOrdinal>& send_length = A.send_length;
  const std::vector<int>& neighbors = A.neighbors;
  const std::vector<GlobalOrdinal>& elements_to_send = A.elements_to_send;

  std::vector<Scalar>& send_buffer = A.send_buffer;

  //
  // first post receives, these are immediate receives
  // Do not wait for result to come, will do that at the
  // wait call below.
  //

  int MPI_MY_TAG = 99;

  std::vector<MPI_Request>& request = A.request;
  int nextReqIndex = 0;

  //
  // Externals are at end of locals
  //

  std::vector<Scalar>& x_coefs = x.coefs;

  MPI_Datatype mpi_dtype = TypeTraits<Scalar>::mpi_type();

  struct recv_args args = { x_coefs, request, recv_length, mpi_dtype, neighbors, local_nrow};

  QLOOP(0, num_neighbors, post_recvs, &args);

#ifdef MINIFE_DEBUG
  os << "launched recvs\n";
#endif

  //
  // Fill up send buffer
  //

  size_t total_to_be_sent = elements_to_send.size();
#ifdef MINIFE_DEBUG
  os << "total_to_be_sent: " << total_to_be_sent << std::endl;
#endif

  for(size_t i=0; i<total_to_be_sent; ++i) {
#ifdef MINIFE_DEBUG
    //expensive index range-check:
    if (elements_to_send[i] < 0 || elements_to_send[i] > x.coefs.size()) {
      os << "error, out-of-range. x.coefs.size()=="<<x.coefs.size()<<", elements_to_send[i]=="<<elements_to_send[i]<<std::endl;
    }
#endif
    send_buffer[i] = x.coefs[elements_to_send[i]];
  }

  //
  // Send to each neighbor
  //

  struct send_args sargs = {send_buffer, send_length, mpi_dtype, neighbors };
  QLOOP(0, num_neighbors, send_func, &sargs);

#ifdef MINIFE_DEBUG
  os << "send to " << num_neighbors << std::endl;
#endif

  //
  // Complete the reads issued above
  //
  /*
  MPI_Status status;
  for(int i=0; i<num_neighbors; ++i) {
    if (MPI_Wait(&request[i], &status) != MPI_SUCCESS) {
      std::cerr << "MPI_Wait error\n"<<std::endl;
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
  }
  */
#ifdef MINIFE_DEBUG
  os << "leaving exchange_externals"<<std::endl;
#endif

//endif HAVE_MPI
#endif
}

#ifdef HAVE_MPI
static std::vector<MPI_Request> exch_ext_requests;
#endif

template<typename MatrixType,
         typename VectorType>
void
begin_exchange_externals(MatrixType& A,
                         VectorType& x)
{
#ifdef HAVE_MPI

  int numprocs = 1, myproc = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myproc);

  if (numprocs < 2) return;

  typedef typename MatrixType::ScalarType Scalar;
  typedef typename MatrixType::LocalOrdinalType LocalOrdinal;
  typedef typename MatrixType::GlobalOrdinalType GlobalOrdinal;

  // Extract Matrix pieces

  int local_nrow = A.rows.size();
  int num_neighbors = A.neighbors.size();
  const std::vector<LocalOrdinal>& recv_length = A.recv_length;
  const std::vector<LocalOrdinal>& send_length = A.send_length;
  const std::vector<int>& neighbors = A.neighbors;
  const std::vector<GlobalOrdinal>& elements_to_send = A.elements_to_send;

  std::vector<Scalar> send_buffer(elements_to_send.size(), 0);

  //
  // first post receives, these are immediate receives
  // Do not wait for result to come, will do that at the
  // wait call below.
  //

  int MPI_MY_TAG = 99;

  exch_ext_requests.resize(num_neighbors);

  //
  // Externals are at end of locals
  //

  std::vector<Scalar>& x_coefs = x.coefs;
  Scalar* x_external = &(x_coefs[local_nrow]);

  MPI_Datatype mpi_dtype = TypeTraits<Scalar>::mpi_type();

  // Post receives first
  for(int i=0; i<num_neighbors; ++i) {
    int n_recv = recv_length[i];
    MPI_Irecv(x_external, n_recv, mpi_dtype, neighbors[i], MPI_MY_TAG,
              MPI_COMM_WORLD, &exch_ext_requests[i]);
    x_external += n_recv;
  }

  //
  // Fill up send buffer
  //

  size_t total_to_be_sent = elements_to_send.size();
  for(size_t i=0; i<total_to_be_sent; ++i) send_buffer[i] = x.coefs[elements_to_send[i]];

  //
  // Send to each neighbor
  //

  Scalar* s_buffer = &send_buffer[0];

  for(int i=0; i<num_neighbors; ++i) {
    int n_send = send_length[i];
    MPI_Send(s_buffer, n_send, mpi_dtype, neighbors[i], MPI_MY_TAG,
             MPI_COMM_WORLD);
    s_buffer += n_send;
  }
#endif
}

inline
void
finish_exchange_externals(int num_neighbors)
{
#ifdef HAVE_MPI
  //
  // Complete the reads issued above
  //

  MPI_Status status;
  for(int i=0; i<num_neighbors; ++i) {
    if (MPI_Wait(&exch_ext_requests[i], &status) != MPI_SUCCESS) {
      std::cerr << "MPI_Wait error\n"<<std::endl;
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
  }

//endif HAVE_MPI
#endif
}

}//namespace miniFE

#endif


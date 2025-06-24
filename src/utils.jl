using LinearAlgebra.BLAS
using LinearAlgebra

function dotu(a::AbstractVector{ComplexF64}, b::AbstractVector{ComplexF64})
  n = length(a)
  return BLAS.dotu(n, a, 1, b, 1)
end

function dotu(a::AbstractVector{ComplexF32}, b::AbstractVector{ComplexF32})
  n = length(a)
  return BLAS.dotu(n, a, 1, b, 1)
end

function dotu(a::Complex, b::Complex)
  return a * b
end

function dotu(a::AbstractVector, b::AbstractVector)
  return dot(a, b)
end

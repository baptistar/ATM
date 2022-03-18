function dxS = InnerProd(A,B,dim)
% InnerProd computes the inner product of matrix A with B along the
% dimension dim.
%
% Inputs: A, B - arbitary matrices
%         dim - index along which to compute inner product

% check dimensions
if size(A, dim) ~= size(B, dim)
    error('Size of A and b dont''t match along specified dimension')
end

% compute inner product
dxS = squeeze(sum( A .* B, dim));

end
function C = OuterProd(A, B)
% OuterProd computes outer product of each row in A (N x d) and
% B (N x d) to produce an (N x d x d) matrix C
%
% Inputs:  A - (N x d) matrix
%          B - (N x d) matrix
% Outputs: C - (N x d x d) matrix

    % find dimensions of A and B
    Asz = size(A);
    Bsz = size(B);

    % check dimensions
    if Asz(1) ~= Bsz(1)
        error('Dimension mismath for number of samples in A and B')
    end

    % compute outer product using elementwise multiplication
    %A_rep = repmat(A,1,1,Bsz(2));
    %B_rep = repmat(reshape(B,Bsz(1),1,Bsz(2)),1,Asz(2),1);
    %C = A_rep.*B_rep;
    C = bsxfun(@times, A, permute(B, [1 3 2]));

end %endFunction
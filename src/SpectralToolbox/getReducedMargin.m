function ReducedMargin = getReducedMargin(multi_idxs)
% function Imarg = getReducedMargin(I)
% Returns the reduced margin of the multi-index set I defined
% by the set of multi-indices i not in I such that
% \forall k \in N^\ast s.t. i_k \neq 0 \implies i - e_k \in I
% where e_k is the k-th Kronecker sequence
% I,Imarg: MultiIndices

if isempty(multi_idxs)
    ReducedMargin = zeros(0,0);%error('multi_idxs should not be empty')
    return
end
d = size(multi_idxs,2);
n = size(multi_idxs,1);
neighbours = repmat(permute(multi_idxs,[1,3,2]),[1,d,1]) + repmat(permute(eye(d),[3,1,2]),[n,1,1]);
neighbours = reshape(neighbours,n*d,d);
Margin = setdiff(neighbours,multi_idxs,'rows');
Margin = unique(Margin,'rows');

cardinal = size(Margin,1);
neighbours = repmat(permute(Margin,[1,3,2]),[1,d,1]) - repmat(permute(eye(d),[3,1,2]),[cardinal,1,1]);
n = size(neighbours,1);

neighbours = reshape(neighbours,n*d,d);
ok = ismember(neighbours,multi_idxs,'rows');
isout = any(neighbours<0,2);
ok = ok | isout;
ok = reshape(ok,[n,d]);
keep = all(ok,2);
ReducedMargin = Margin(keep,:);

end

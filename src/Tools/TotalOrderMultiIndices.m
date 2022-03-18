function multi_idxs = TotalOrderMultiIndices(order_list)
% TotalOrderMultiIndices: Compute indices of total degree 
% polynomial expansion. Output is an (ncoeff x d) matrix
% where 0 corresponds to the constant function
%
% Inputs:  order_list - (d x 1) max order for each dimension
% Outputs: multi_idxs - (ncoeff x d) indices

    % return multi_idxs to be all zeros if order_list is empty (i.e., d=0)
    if isempty(order_list)
        multi_idxs = zeros(1,0);
        return
    end

	% determine dimension and max_order
	d = length(order_list);
	max_order = max(order_list);
	
    % initialize multi_index with zeros
	midxs_new = zeros(1,d); 
    multi_idxs = midxs_new;

	% initialize midxs_old_set
	if sum(midxs_new) < max_order
		midxs_old_set = midxs_new;
	else
		midxs_old_set = zeros(0,d);
	end

	% generate higher order multi-indices
	for i=1:max_order

		% initialize empty set of starting multi-indices
		midxs_new_set = zeros(0,d);
		
		% extract each multi_idx in midxs_old_set
		for j=1:size(midxs_old_set,1)
			midxs_old_j = midxs_old_set(j,:);

			% expand index set along each direction
			for k=1:d

				% if allowable, add new multi_idx
				if midxs_old_j(k) < order_list(k)
					midx_new = midxs_old_j;
					midx_new(k) = midx_new(k) + 1;
					multi_idxs = [multi_idxs; midx_new];

					% if boundary of orders isn't add, expand
					% in the next iteration by adding to set
					if sum(midx_new) < max_order
						midxs_new_set = [midxs_new_set; midx_new];
					end
				
				end

			end

		end
		
		% overwrite midxs_old_set for next iteration
		midxs_old_set = midxs_new_set;

    end
    
    % remove duplicates in multi_idxs
    multi_idxs = unique(multi_idxs,'rows'); 

end %endFunction
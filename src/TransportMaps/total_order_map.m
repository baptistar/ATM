function S = total_order_map(comps, basis, order)

    % find maximum size
    d = max(comps);

    % repeat basis if necessary
    if length(basis) == 1
        basis = repmat({basis},1,d);
    end
    % repeat order if necessary
    if length(order) == 1
        order = repmat(order,1,d);
    end

    % define each component
    S = cell(d,1);
    for k = comps
        S{k} = total_order_component(k, basis(1:k), order(k));
    end
    
end

function Sk = total_order_component(k, basis, order)

    % define initial m_idx
    m_idx = TotalOrderMultiIndices(order*ones(1,k));
    
    % define polynomial and integrated function
    P = ParametericPoly(basis, m_idx);
    Sk = IntegratedPositiveFunction(P);
    Sk = Sk.set_coeff(zeros(1,size(m_idx,1)));

end

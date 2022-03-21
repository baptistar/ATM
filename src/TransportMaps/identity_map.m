function S = identity_map(comps, basis)

    % find maximum size
    d = max(comps);

    % repeat basis if necessary
    if length(basis) == 1
        basis = repmat({basis},1,d);
    end

    % define each component
    S = cell(d,1);
    for k = comps
        S{k} = identity_component(k, basis(1:k));
    end
    
end

function Sk = identity_component(k, basis)

    % define initial m_idx
    m_idx = zeros(1,k);

    % define polynomial and integrated function
    P = ParametericPoly(basis, m_idx);
    Sk = IntegratedPositiveFunction(P);
    Sk = Sk.set_coeff(0);

end

function [lower_set,reduced_margin] = UpdateReducedMargin(lower_set,reduced_margin,ind)

d = size(lower_set,2);
new_indx = reduced_margin(ind,:);
lower_set = [lower_set ; reduced_margin(ind,:)];
reduced_margin(ind,:) = [];

% Update the reduced margin
candidate = repmat(new_indx,d,1) + eye(d);
ok = false(d,1);
for i=1:d
    parents_of_candidate = repmat(candidate(i,:),d,1) - eye(d);
    parents_of_candidate(any(parents_of_candidate==-1,2),:) = [];
    ok(i) = all(ismember(parents_of_candidate,lower_set,'rows'));
end
candidate = candidate(ok,:);

% Add the candidates to the reduced margin
reduced_margin = [reduced_margin ; candidate];

end

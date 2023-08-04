index  = ismember(model_gene, data_gene, 'rows');
result1 = model_gene(index, :)

result = intersect(model_gene, data_gene, 'rows');

a=1
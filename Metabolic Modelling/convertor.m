
clear all
my_data = readtable('gene_ids.xlsx');
ref_data = readtable('gene-formats.xlsx');

for i=1:size(my_data, 1)
    gene_id = my_data(i, 1);
    aa = find(ref_data(:, 3) = gene_id)
end
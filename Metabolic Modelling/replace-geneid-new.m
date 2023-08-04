human.grRules = grRules;
human.genes = genes;
human.rxnGeneMat = rxnGeneMat;
%[grRules, genes, rxnGeneMat] = translateGrRules(human.grRules,'ENSG','Name', 'delete');
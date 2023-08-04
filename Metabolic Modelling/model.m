load('D:\Second-year\Human-GEM\model\Human-GEM.mat');
human1 = ravenCobraWrapper(ihuman);
human1 = creategrRulesField(human1);
for i=1:length(human1.subSystems)                     % fix type of field
   human1.subSystems{i} = human1.subSystems{i}{1};
end
save('human1.mat', 'human1');

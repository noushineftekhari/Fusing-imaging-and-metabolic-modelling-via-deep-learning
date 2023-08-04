gene_exp_healthy = xlsread('D:\Second-year\RNA-seq-OV\Gtex-noiso-ENSG.xlsx','Sheet1');
'A1:CL1', 'A1:A33530');
%gene_exp = xlsread('OV_D.xlsx','Sheet1','B1:KX1','A2:A20532');
%patient_ids = xlsread('OV_D.xlsx','Sheet1','A1','B1:KX1');
%gene_ids = xlsread('OV_D.xlsx','Sheet1','A2:A20532');
patient_ids = xlsread('D:\Second-year\RNA-seq-OV\TCGA-ov-toil-noiso-ENSG.xlsx','Sheet1','A1','B1:PL1');
gene_ids= xlsread('D:\Second-year\RNA-seq-OV\TCGA-ov-toil-noiso-ENSG.xlsx','Sheet2','A1','A2:A33767');
x=xlsread('D:\Second-year\RNA-seq-OV\TCGA-ov-toil-noiso-ENSG.xlsx','Sheet2');
patient_ids = xlsread('D:\Second-year\RNA-seq-OV\TCGA-ov-toil-noiso-ENSG.xlsx','Sheet3','A1','B1:PL1');

gene_exp = transpose(gene_exp_healthy);
data= gene_exp (3:end,: );
num1 = xlsread('patient_ids.xlsx');
T = readtable('patient_ids.xlsx');
 [~,txtData]  = xlsread('patient_ids_TCGA.xlsx','A1:PL1');
patient_ids = transpose(txtData);
[~,gene_ids]  = xlsread('gene_id_TCGA.xlsx','A2:A33530');
gene_ids=transpose(gene_ids);
% mean healthy
mean=mean(gene_exp_healthy)
C = num2cell(mean);
Con = cat(1,gene_ids,C)
gene_exp_Gtex = data
gene_ids_Gtex=gene_ids;

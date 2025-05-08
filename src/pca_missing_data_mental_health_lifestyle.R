#!/usr/bin/env Rscript

library(missMDA)
library(FactoMineR)

data = read.table('tmp/ukbb_mental_health_binary.csv', sep=',', header=TRUE)
#nb = estim_ncpPCA(data[,2:240],ncp.max=35)
#res.comp = imputePCA(data[,2:240], ncp=nb$ncp)
for (ncp in c(2,5)){
    res.comp = imputePCA(data[,3:241], ncp=ncp)
    res.pca = PCA(res.comp$completeObs)

    # save results of estimated values
    df = data.frame(res.comp$fittedX)
    colnames(df) = colnames(data[,3:241])
    write.table(df, paste('tmp/ukbb_mental_health_binary_imputed_pca_ncp', ncp, '.csv', sep=''), quote=FALSE, sep=',', row.names=FALSE)

    # save results on eig
    write.table(res.pca$eig, paste('tmp/ukbb_mental_health_binary_pca_eig_ncp', ncp, '.csv', sep=''), quote=FALSE, sep=',', row.names=FALSE)

    # save results on dim desc
    res_dim = dimdesc(res.pca, axes=1:5, proba=0.05)
    write.table(res_dim$Dim.1$quanti, paste('tmp/ukbb_mental_health_binary_pca_dim1_corr_ncp', ncp, '.csv', sep=''), quote=FALSE, sep=',', row.names=TRUE)
    write.table(res_dim$Dim.2$quanti, paste('tmp/ukbb_mental_health_binary_pca_dim2_corr_ncp', ncp, '.csv', sep=''), quote=FALSE, sep=',', row.names=TRUE)
    write.table(res_dim$Dim.3$quanti, paste('tmp/ukbb_mental_health_binary_pca_dim3_corr_ncp', ncp, '.csv', sep=''), quote=FALSE, sep=',', row.names=TRUE)
    write.table(res_dim$Dim.4$quanti, paste('tmp/ukbb_mental_health_binary_pca_dim4_corr_ncp', ncp, '.csv', sep=''), quote=FALSE, sep=',', row.names=TRUE)
    write.table(res_dim$Dim.5$quanti, paste('tmp/ukbb_mental_health_binary_pca_dim5_corr_ncp', ncp, '.csv', sep=''), quote=FALSE, sep=',', row.names=TRUE)

    # save UKBB data in the new dimensions
    write.table(res.pca$ind$coord, paste('tmp/ukbb_mental_health_binary_pca_coord_dim5_ncp', ncp, '.csv', sep=''), quote=FALSE, sep=',', row.names=FALSE)
}

data = read.table('tmp/ukbb_lifestyle_binary.csv', sep=',', header=TRUE)
#nb = estim_ncpPCA(data[,2:240],ncp.max=35)
#res.comp = imputePCA(data[,2:240], ncp=nb$ncp)
for (ncp in 2:5){
    res.comp = imputePCA(data[,3:183], ncp=ncp)
    res.pca = PCA(res.comp$completeObs)

    # save results of estimated values
    df = data.frame(res.comp$fittedX)
    colnames(df) = colnames(data[,3:183])
    write.table(df, paste('tmp/ukbb_lifestyle_binary_imputed_pca_ncp', ncp, '.csv', sep=''), quote=FALSE, sep=',', row.names=FALSE)

    # save results on eig
    write.table(res.pca$eig, paste('tmp/ukbb_lifestyle_binary_pca_eig_ncp', ncp, '.csv', sep=''), quote=FALSE, sep=',', row.names=FALSE)

    # save results on dim desc
    res_dim = dimdesc(res.pca, axes=1:5, proba=0.05)
    write.table(res_dim$Dim.1$quanti, paste('tmp/ukbb_lifestyle_binary_pca_dim1_corr_ncp', ncp, '.csv', sep=''), quote=FALSE, sep=',', row.names=TRUE)
    write.table(res_dim$Dim.2$quanti, paste('tmp/ukbb_lifestyle_binary_pca_dim2_corr_ncp', ncp, '.csv', sep=''), quote=FALSE, sep=',', row.names=TRUE)
    write.table(res_dim$Dim.3$quanti, paste('tmp/ukbb_lifestyle_binary_pca_dim3_corr_ncp', ncp, '.csv', sep=''), quote=FALSE, sep=',', row.names=TRUE)
    write.table(res_dim$Dim.4$quanti, paste('tmp/ukbb_lifestyle_binary_pca_dim4_corr_ncp', ncp, '.csv', sep=''), quote=FALSE, sep=',', row.names=TRUE)
    write.table(res_dim$Dim.5$quanti, paste('tmp/ukbb_lifestyle_binary_pca_dim5_corr_ncp', ncp, '.csv', sep=''), quote=FALSE, sep=',', row.names=TRUE)

    # save UKBB data in the new dimensions
    write.table(res.pca$ind$coord, paste('tmp/ukbb_lifestyle_binary_pca_coord_dim5_ncp', ncp, '.csv', sep=''), quote=FALSE, sep=',', row.names=FALSE)
}
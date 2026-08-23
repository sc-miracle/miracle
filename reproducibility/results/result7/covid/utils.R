# setwd('../')
library(Seurat)
library(SeuratDisk)
library(Signac)
library(future)
library(EnsDb.Hsapiens.v86)
library(BSgenome.Hsapiens.UCSC.hg38)
library(dplyr)
library(ggplot2)
library(Matrix)
library(purrr)
library(stringr)
library(GenomicRanges)
library(RcppTOML)
library(ymlthis)
library(argparse)
library(RColorBrewer)
library(hdf5r)
set.seed(1234)
plan("multicore", workers = 4)
options(future.globals.maxSize = 100 * 1024^3) # for 100 Gb RAM
options(future.seed = T)

library(argparse)


pj <- file.path


prt <- function(...) {
    cat(paste0(..., "\n"))
}



mkdir <- function(directory, remove_old = F) {
    if (remove_old) {
        if (dir.exists(directory)) {
             prt("Removing directory ", directory)
             unlink(directory, recursive = T)
        }
    }
    if (!dir.exists(directory)) {
        dir.create(directory, recursive = T)
    }
}


mkdirs <- function(directories, remove_old = F) {
    for (directory in directories) {
        mkdir(directory, remove_old = remove_old)
    }
}
gen_atac <- function(frag_path, min_cells = 5) {
    # call peaks using MACS2
    system(paste0("tabix -f -p bed ", frag_path))
    frags <- CreateFragmentObject(frag_path)
    peaks <- CallPeaks(frags)
    peaks@seqnames
    # remove peaks on non-autosomes and in genomic blacklist regions
    peaks <- keepStandardChromosomes(peaks, pruning.mode = "coarse")
    peaks <- peaks[!(peaks@seqnames %in% c("chrX", "chrY"))]
    peaks <- subsetByOverlaps(x = peaks, ranges = blacklist_hg38_unified, invert = TRUE)
    # quantify counts in each peak
    atac_counts <- FeatureMatrix(
        fragments = frags,
        features = peaks
    )
    # # add in the atac-seq data, only use peaks in standard chromosomes
    # grange <- StringToGRanges(rownames(atac_counts))
    # grange_use <- seqnames(grange) %in% standardChromosomes(grange)
    # atac_counts <- atac_counts[as.vector(grange_use), ]
    # get gene annotations for hg38
    annotation <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86)
    seqlevelsStyle(annotation) <- "UCSC"
    genome(annotation) <- "hg38"
    # create atac assay and add it to the object
    atac_assay <- CreateChromatinAssay(
        counts = atac_counts,
        min.cells = min_cells,
        genome = 'hg38',
        fragments = frags,
        annotation = annotation
    )
    atac <- CreateSeuratObject(
        counts = atac_assay,
        assay = 'atac',
    )
    atac <- NucleosomeSignal(atac)
    atac <- TSSEnrichment(atac)
    return(atac)
}


gen_rna <- function(rna_counts, min_cells = 3) {
    rna <- CreateSeuratObject(
        counts = rna_counts,
        min.cells = min_cells,
        assay = "rna"
    )
    rna[["percent.mt"]] <- PercentageFeatureSet(rna, pattern = "^MT-")
    return(rna)
}


remove_sparse_genes <- function(obj, assay = "rna", min_cell_percent = 1, kept_genes = NULL) {
    assay_ <- DefaultAssay(obj)
    DefaultAssay(obj) <- assay
    min_cells <- 0.01 * min_cell_percent * ncol(obj)
    mask <- rowSums(obj[[assay]]@counts > 0) > min_cells & rowSums(obj[[assay]]@counts) > 2 * min_cells
    feats <- rownames(obj[[assay]]@counts)[mask]
    if (!is.null(kept_genes)) {
        feats <- union(feats, kept_genes)
    }
    obj <- subset(obj, features = feats)
    DefaultAssay(obj) <- assay_
    return(obj)
}


gen_adt <- function(adt_counts) {
    # rename features
    feat <- unlist(map(rownames(adt_counts), tolower))
    feat <- unlist(map(feat, gsub, pattern = "-|_|\\(|\\)|/", replacement = "."))
    rownames(adt_counts) <- feat
    if (length(grep("^hla.dr$|^hla.dp$|^hla.dq$|^hla.dr.dp.dq$", rownames(adt_counts))) > 1) {
        hla_index <- grep("^hla.dr$|^hla.dp$|^hla.dq$|^hla.dr.dp.dq$", rownames(adt_counts))
        for (i in 1:length(hla_index)) {
            if (i == 1) {
                hla_drdpdq <- adt_counts[hla_index[i],]
            } else {
                hla_drdpdq <- hla_drdpdq + adt_counts[hla_index[i],]
            }
        }
        adt_counts <- adt_counts[-grep("^hla.dr$|^hla.dp$|^hla.dq$|^hla.dr.dp.dq$", rownames(adt_counts)), ]
        adt_counts <- bind_rows(adt_counts, hla_drdpdq)
    }
    feat <- unlist(rownames(adt_counts))
    feat <- unlist(map(feat, gsub, pattern = "^cd3$", replacement = "cd3.1"))
    feat <- unlist(map(feat, gsub, pattern = "^cd4$", replacement = "cd4.2"))
    feat <- unlist(map(feat, gsub, pattern = "^cd11b$", replacement = "cd11b.2"))
    feat <- unlist(map(feat, gsub, pattern = "^cd26$", replacement = "cd26.2"))
    feat <- unlist(map(feat, gsub, pattern = "^cd38$", replacement = "cd38.2"))
    feat <- unlist(map(feat, gsub, pattern = "^cd56$", replacement = "cd56.1"))
    feat <- unlist(map(feat, gsub, pattern = "^cd56.ncam.$", replacement = "cd56.1"))
    feat <- unlist(map(feat, gsub, pattern = "^cd56.ncam.recombinant$", replacement = "cd56.2"))
    feat <- unlist(map(feat, gsub, pattern = "^cd57.recombinant$", replacement = "cd57"))
    feat <- unlist(map(feat, gsub, pattern = "^cd90.thy1.$", replacement = "cd90"))
    feat <- unlist(map(feat, gsub, pattern = "^cd112.nectin.2.$", replacement = "cd112"))
    feat <- unlist(map(feat, gsub, pattern = "^cd117.c.kit.$", replacement = "cd117"))
    feat <- unlist(map(feat, gsub, pattern = "^cd138.1.syndecan.1.$", replacement = "cd138.1"))
    feat <- unlist(map(feat, gsub, pattern = "^cd155.pvr.$", replacement = "cd155"))
    feat <- unlist(map(feat, gsub, pattern = "^cd269.bcma.$", replacement = "cd269"))
    feat <- unlist(map(feat, gsub, pattern = "^clec2$", replacement = "clec1b"))
    feat <- unlist(map(feat, gsub, pattern = "^cadherin11$", replacement = "cadherin"))
    feat <- unlist(map(feat, gsub, pattern = "^folate.receptor$", replacement = "folate"))
    feat <- unlist(map(feat, gsub, pattern = "^notch.1$", replacement = "notch1"))
    feat <- unlist(map(feat, gsub, pattern = "^notch.2$", replacement = "notch3"))
    feat <- unlist(map(feat, gsub, pattern = "^tcr.a.b$", replacement = "tcrab"))
    feat <- unlist(map(feat, gsub, pattern = "^tcr.2$", replacement = "tcrab"))
    feat <- unlist(map(feat, gsub, pattern = "^tcr.g.d$", replacement = "tcrgd"))
    feat <- unlist(map(feat, gsub, pattern = "^tcr.1$", replacement = "tcrgd"))
    feat <- unlist(map(feat, gsub, pattern = "^tcr.va7.2$", replacement = "tcr.v.7.2"))
    feat <- unlist(map(feat, gsub, pattern = "^tcr.va24.ja18$", replacement = "tcr.v.24.j.18"))
    feat <- unlist(map(feat, gsub, pattern = "^vegfr.3$", replacement = "vegfr3"))
    feat <- unlist(map(feat, gsub, pattern = "^ccr5$", replacement = "cd195"))
    feat <- unlist(map(feat, gsub, pattern = "^ccr7$", replacement = "cd197"))
    feat <- unlist(map(feat, gsub, pattern = "^tcr.ab$", replacement = "tcrab"))
    feat <- unlist(map(feat, gsub, pattern = "^integrin.b7$", replacement = "integrin.7"))
    feat <- unlist(map(feat, gsub, pattern = "^hla.dr$", replacement = "hla.drdpdq"))
    feat <- unlist(map(feat, gsub, pattern = "^hla.dq$", replacement = "hla.drdpdq"))
    feat <- unlist(map(feat, gsub, pattern = "^hla.dp$", replacement = "hla.drdpdq"))
    feat <- unlist(map(feat, gsub, pattern = "^hla.dr.dp.dq$", replacement = "hla.drdpdq"))
    feat <- unlist(map(feat, gsub, pattern = "^tcr.vd2$", replacement = "tcr.v.2"))
    feat <- unlist(map(feat, gsub, pattern = "^cd275$", replacement = "cd275.2"))
    feat <- unlist(map(feat, gsub, pattern = "^cd45$", replacement = "cd45.2"))
    feat <- unlist(map(feat, gsub, pattern = "^cd44$", replacement = "cd44.1"))
    feat <- unlist(map(feat, gsub, pattern = "^pdcd1lg2$", replacement = "cd273"))
    feat <- unlist(map(feat, gsub, pattern = "^icoslg$", replacement = "cd275"))
    feat <- unlist(map(feat, gsub, pattern = "^itgam$", replacement = "cd11b.2"))
    feat <- unlist(map(feat, gsub, pattern = "^ox40l$", replacement = "cd252"))
    feat <- unlist(map(feat, gsub, pattern = "^tnfsf9$", replacement = "cd137l"))
    feat <- unlist(map(feat, gsub, pattern = "^pvr$", replacement = "cd155"))
    feat <- unlist(map(feat, gsub, pattern = "^nectin2$", replacement = "cd112"))
    feat <- unlist(map(feat, gsub, pattern = "^tnfrsf8$", replacement = "cd30"))
    feat <- unlist(map(feat, gsub, pattern = "^cd40lg$", replacement = "cd154"))
    feat <- unlist(map(feat, gsub, pattern = "^itgax$", replacement = "cd11c"))
    feat <- unlist(map(feat, gsub, pattern = "^tnfrsf17$", replacement = "cd269"))
    feat <- unlist(map(feat, gsub, pattern = "^hla.abc$", replacement = "hla.a.b.c"))
    feat <- unlist(map(feat, gsub, pattern = "^thy1$", replacement = "cd90"))
    feat <- unlist(map(feat, gsub, pattern = "^kit$", replacement = "cd117"))
    feat <- unlist(map(feat, gsub, pattern = "^mme$", replacement = "cd10"))
    feat <- unlist(map(feat, gsub, pattern = "^itga6$", replacement = "cd49f"))
    feat <- unlist(map(feat, gsub, pattern = "^ccr4$", replacement = "cd194"))
    feat <- unlist(map(feat, gsub, pattern = "^pd1$", replacement = "cd279"))
    feat <- unlist(map(feat, gsub, pattern = "^ncr1$", replacement = "cd335"))
    feat <- unlist(map(feat, gsub, pattern = "^ptgdr2$", replacement = "cd294"))
    feat <- unlist(map(feat, gsub, pattern = "^epcam$", replacement = "cd326"))
    feat <- unlist(map(feat, gsub, pattern = "^pecam1$", replacement = "cd31"))
    feat <- unlist(map(feat, gsub, pattern = "^mcam$", replacement = "cd146"))
    feat <- unlist(map(feat, gsub, pattern = "^cdh1$", replacement = "cd324"))
    feat <- unlist(map(feat, gsub, pattern = "^tcrg.d$", replacement = "tcrgd"))
    feat <- unlist(map(feat, gsub, pattern = "^cxcr3$", replacement = "cd183"))
    feat <- unlist(map(feat, gsub, pattern = "^fcgr2a$", replacement = "cd32"))
    feat <- unlist(map(feat, gsub, pattern = "^ccr6$", replacement = "cd196"))
    feat <- unlist(map(feat, gsub, pattern = "^cxcr5$", replacement = "cd185"))
    feat <- unlist(map(feat, gsub, pattern = "^itgae$", replacement = "cd103"))
    feat <- unlist(map(feat, gsub, pattern = "^ctla4$", replacement = "cd152"))
    feat <- unlist(map(feat, gsub, pattern = "^lag3$", replacement = "cd223"))
    feat <- unlist(map(feat, gsub, pattern = "^lamp1$", replacement = "cd107a"))
    feat <- unlist(map(feat, gsub, pattern = "^fas$", replacement = "cd95"))
    feat <- unlist(map(feat, gsub, pattern = "^klrk1$", replacement = "cd314"))
    feat <- unlist(map(feat, gsub, pattern = "^ceacam8$", replacement = "cd66b"))
    feat <- unlist(map(feat, gsub, pattern = "^cr1$", replacement = "cd35"))
    feat <- unlist(map(feat, gsub, pattern = "^b3gat1$", replacement = "cd57"))
    feat <- unlist(map(feat, gsub, pattern = "^havcr2$", replacement = "cd366"))
    feat <- unlist(map(feat, gsub, pattern = "^btla$", replacement = "cd272"))
    feat <- unlist(map(feat, gsub, pattern = "^icos$", replacement = "cd278"))
    feat <- unlist(map(feat, gsub, pattern = "^entpd1$", replacement = "cd39"))
    feat <- unlist(map(feat, gsub, pattern = "^faslg$", replacement = "cd178"))
    feat <- unlist(map(feat, gsub, pattern = "^itgal$", replacement = "cd11a"))
    feat <- unlist(map(feat, gsub, pattern = "^iga$", replacement = "cd79a"))
    feat <- unlist(map(feat, gsub, pattern = "^ceacam1.5.6$", replacement = "cd66a.c.e"))
    feat <- unlist(map(feat, gsub, pattern = "^mmr$", replacement = "cd206"))
    feat <- unlist(map(feat, gsub, pattern = "^siglec1$", replacement = "cd169"))
    feat <- unlist(map(feat, gsub, pattern = "^clec9a$", replacement = "cd370"))
    feat <- unlist(map(feat, gsub, pattern = "^itgb7$", replacement = "integrin.7"))
    feat <- unlist(map(feat, gsub, pattern = "^baffr$", replacement = "cd268"))
    feat <- unlist(map(feat, gsub, pattern = "^icam1$", replacement = "cd54"))
    feat <- unlist(map(feat, gsub, pattern = "^selp$", replacement = "cd62p"))
    feat <- unlist(map(feat, gsub, pattern = "^tcr$", replacement = "tcrab"))
    feat <- unlist(map(feat, gsub, pattern = "^vcam1$", replacement = "cd106"))
    feat <- unlist(map(feat, gsub, pattern = "^il2rb$", replacement = "cd122"))
    feat <- unlist(map(feat, gsub, pattern = "^taci$", replacement = "cd267"))
    feat <- unlist(map(feat, gsub, pattern = "^itga2b$", replacement = "cd41"))
    feat <- unlist(map(feat, gsub, pattern = "^tnfrsf9$", replacement = "cd137"))
    feat <- unlist(map(feat, gsub, pattern = "^rankl$", replacement = "cd254"))
    feat <- unlist(map(feat, gsub, pattern = "^gitr$", replacement = "cd357"))
    feat <- unlist(map(feat, gsub, pattern = "^kdr$", replacement = "cd309"))
    feat <- unlist(map(feat, gsub, pattern = "^il4r$", replacement = "cd124"))
    feat <- unlist(map(feat, gsub, pattern = "^cxcr4$", replacement = "cd184"))
    feat <- unlist(map(feat, gsub, pattern = "^itgb1$", replacement = "cd29"))
    feat <- unlist(map(feat, gsub, pattern = "^itga2$", replacement = "cd49b"))
    feat <- unlist(map(feat, gsub, pattern = "^slc3a2$", replacement = "cd98"))
    feat <- unlist(map(feat, gsub, pattern = "^itgb2$", replacement = "cd18"))
    feat <- unlist(map(feat, gsub, pattern = "^il7r$", replacement = "cd127"))
    feat <- unlist(map(feat, gsub, pattern = "^dpp4$", replacement = "cd26"))
    feat <- unlist(map(feat, gsub, pattern = "^ccr3$", replacement = "cd193"))
    feat <- unlist(map(feat, gsub, pattern = "^msr1$", replacement = "cd204"))
    feat <- unlist(map(feat, gsub, pattern = "^cdh5$", replacement = "cd144"))
    feat <- unlist(map(feat, gsub, pattern = "^langerin$", replacement = "cd207"))
    feat <- unlist(map(feat, gsub, pattern = "^itga4$", replacement = "cd49d"))
    feat <- unlist(map(feat, gsub, pattern = "^nt5e$", replacement = "cd73"))
    feat <- unlist(map(feat, gsub, pattern = "^tcr.vg2$", replacement = "tcr.v.2"))
    feat <- unlist(map(feat, gsub, pattern = "^tcr.vg9$", replacement = "tcr.v.9"))
    feat <- unlist(map(feat, gsub, pattern = "^lair1$", replacement = "cd305"))
    feat <- unlist(map(feat, gsub, pattern = "^olr1$", replacement = "lox.1"))
    feat <- unlist(map(feat, gsub, pattern = "^prom1$", replacement = "cd133.2"))
    feat <- unlist(map(feat, gsub, pattern = "^kir3dl1$", replacement = "cd158e1"))
    feat <- unlist(map(feat, gsub, pattern = "^kir2dl5a$", replacement = "cd158f"))
    feat <- unlist(map(feat, gsub, pattern = "^ncr3$", replacement = "cd337"))
    feat <- unlist(map(feat, gsub, pattern = "^ncr2$", replacement = "cd336"))
    feat <- unlist(map(feat, gsub, pattern = "^fcrl4$", replacement = "cd307d"))
    feat <- unlist(map(feat, gsub, pattern = "^fcrl5$", replacement = "cd307e"))
    feat <- unlist(map(feat, gsub, pattern = "^slamf7$", replacement = "cd319"))
    feat <- unlist(map(feat, gsub, pattern = "^sdc1$", replacement = "cd138.1"))
    feat <- unlist(map(feat, gsub, pattern = "^baff$", replacement = "cd257"))
    feat <- unlist(map(feat, gsub, pattern = "^klrd1$", replacement = "cd94"))
    feat <- unlist(map(feat, gsub, pattern = "^slamf1$", replacement = "cd150"))
    feat <- unlist(map(feat, gsub, pattern = "^lilrb1$", replacement = "cd85j"))
    feat <- unlist(map(feat, gsub, pattern = "^fcer2$", replacement = "cd23"))
    feat <- unlist(map(feat, gsub, pattern = "^iglambda$", replacement = "ig.light.chain.l"))
    feat <- unlist(map(feat, gsub, pattern = "^igkappa$", replacement = "ig.light.chain.k"))
    feat <- unlist(map(feat, gsub, pattern = "^siglec7$", replacement = "cd328"))
    feat <- unlist(map(feat, gsub, pattern = "^tcr.vb.13.1$", replacement = "tcr.v.13.1"))
    feat <- unlist(map(feat, gsub, pattern = "^il21r$", replacement = "cd360"))
    feat <- unlist(map(feat, gsub, pattern = "^c5ar1$", replacement = "cd88"))
    feat <- unlist(map(feat, gsub, pattern = "^ggt1$", replacement = "cd224"))
    feat <- unlist(map(feat, gsub, pattern = "^light$", replacement = "cd258"))
    feat <- unlist(map(feat, gsub, pattern = "^cd275$", replacement = "cd275.1"))
    feat <- unlist(map(feat, gsub, pattern = "^cd26$", replacement = "cd26.1"))
    # feat <- unlist(map(feat, gsub, pattern = "^igg1.k.isotype.control$", replacement = "rat.igg1k.isotypectrl"))
    rownames(adt_counts) <- feat
    # remove features
    if (length(grep("igg", rownames(adt_counts))) != 0) {
        adt_counts <- adt_counts[-grep("igg", rownames(adt_counts)), ]
    }
    # create adt object
    adt <- CreateSeuratObject(
      counts = adt_counts,
      assay = "adt"
    )
    return(adt)
}


preprocess <- function(output_dir, atac = NULL, rna = NULL, adt = NULL) {
    # preprocess and save data
    if (!is.null(atac)) {
        atac <- RunTFIDF(atac) %>%
                FindTopFeatures(min.cutoff = "q0")
        SaveH5Seurat(atac, pj(output_dir, "atac.h5seurat"), overwrite = TRUE)
    }

    if (!is.null(rna)) {
        rna <- NormalizeData(rna) %>%
               FindVariableFeatures(nfeatures = 4000) %>%
               ScaleData()
        SaveH5Seurat(rna, pj(output_dir, "rna.h5seurat"), overwrite = TRUE)
    }

    if (!is.null(adt)) {
        VariableFeatures(adt) <- rownames(adt)
        adt <- NormalizeData(adt, normalization.method = "CLR", margin = 2) %>%
               ScaleData()
        SaveH5Seurat(adt, pj(output_dir, "adt.h5seurat"), overwrite = TRUE)
    }
}


get_adt_genes <- function(file_path = "configs/adt_rna_correspondence.csv") {
    adt_genes_raw <- read.csv(file_path, sep = "\t")[["symbol"]]
    adt_genes <- vector()
    for (gene in adt_genes_raw) {
        if (gene %in% c("not_found", "")) {
            next
        } else if (grepl(",", gene)) {
            adt_genes <- c(adt_genes, strsplit(gene, split = ",")[[1]])
        } else {
            adt_genes <- c(adt_genes, gene)
        }
    }
    return(unique(adt_genes))
}


plt_size <- function(w, h) {
     options(repr.plot.width = w, repr.plot.height = h)
}


dim_plot <- function(obj, w, h, reduction = NULL, split.by = NULL, group.by = NULL,
    label = F, repel = F, label.size = 4, pt.size = NULL, order = NULL, shuffle = T, cols = NULL,
    save_path = NULL, legend = T, title = NULL, display = T, no_axes = F, return_plt = F,
    border = F, raster = F, rater_dpi = 250, ncol=1) {

    plt_size(w = w, h = h)
    plt <- DimPlot(obj, ncol=ncol, reduction = reduction, split.by = split.by, group.by = group.by,
    label = label, repel = repel, label.size = label.size, pt.size = pt.size, shuffle = shuffle,
    order = order, cols = cols, raster = raster, raster.dpi = c(rater_dpi, rater_dpi))
    if (!legend) {
        plt <- plt + NoLegend()
    }

    if (!is.null(title)) {
        plt <- plt + ggtitle(title)
        title_margin <- 5
    } else {
        plt <- plt + theme(plot.title = element_blank())
        title_margin <- 0
    }

    if (no_axes) {
        plt <- plt + NoAxes()
    }

    if (border) {
        plt <- plt + theme(panel.border = element_rect(color = "black", size = 1),
                           axis.ticks.length = unit(0, "pt"), plot.margin = margin(title_margin, 0, 0, 0))
    }

    if (!is.null(save_path)) {
        ggsave(plot = plt, file = paste0(save_path, ".png"), width = w, height = h, limitsize = F)
        ggsave(plot = plt, file = paste0(save_path, ".pdf"), width = w, height = h, limitsize = F)
    }

    if (display) {
        plt
    }

    if (return_plt) {
        return(plt)
    }
}


# https://mokole.com/palette.html
# col_9 <- brewer.pal(n = 9, name = "Set1")

col_34 <- c("#696969", "#228b22", "#7f0000", "#808000", "#483d8b", "#008080", "#cd853f", "#000080", "#9acd32",
            "#32cd32", "#7f007f", "#8fbc8f", "#b03060", "#d2b48c", "#ff0000", "#00ced1", "#ff8c00", "#00ff00",
            "#00fa9a", "#8a2be2", "#dc143c", "#00bfff", "#0000ff", "#f08080", "#da70d6", "#b0c4de", "#ff00ff",
            "#f0e68c", "#ffff54", "#6495ed", "#ff1493", "#7b68ee", "#7fffd4", "#ffc0cb")


col_32 <- c("#808080", "#228b22", "#7f0000", "#808000", "#483d8b", "#008b8b", "#000080", "#d2691e", "#32cd32",
            "#7f007f", "#8fbc8f", "#b03060", "#ff4500", "#ffa500", "#00ff00", "#00fa9a", "#8a2be2", "#dc143c",
            "#00ffff", "#00bfff", "#0000ff", "#adff2f", "#da70d6", "#ff00ff", "#1e90ff", "#fa8072", "#ffff54",
            "#b0e0e6", "#ff1493", "#7b68ee", "#ffdead", "#ffb6c1")


col_27 <- c("#8b4513", "#6b8e23", "#483d8b", "#bc8f8f", "#008080", "#000080", "#daa520", "#8fbc8f", "#8b008b",
            "#b03060", "#ff0000", "#00ff00", "#00fa9a", "#8a2be2", "#dc143c", "#00ffff", "#00bfff", "#0000ff",
            "#adff2f", "#ff7f50", "#ff00ff", "#1e90ff", "#f0e68c", "#ffff54", "#add8e6", "#ff1493", "#ee82ee")

# col_13_ <- c("#00ff00", "#ff0000", "#0000ff", "#ffff00", "#ff69b4", "#00ffff", "#ff00ff", "#008000", "#6495ed",  "#4b0082", "#eee8aa", "#2f4f4f", "#8b4513")
col_13 <- c("#8FC36D", "#f54646", "#4472c4", "#fff300", "#ff69b4", "#ff00ff", "#14e6e6", "#008000", "#82B4ed",  "#D4aaff", "#eee8aa", "#2f4f4f", "#ad6800")


# col_9  <- c("#ff4500", "#006400", "#0000ff", "#ffd700", "#ff1493", "#00ffff", "#4169e1", "#00ff00", "#bc8f8f")

# col_8 <- c("#00ff00", "#ff0000", "#0000ff", "#006400", "#c71585", "#00ffff", "#1e90ff", "#ffd700")
col_8 <- c("#8FC36D", "#f54646", "#4472c4", "#ff00ff", "#82B4ed", "#D4aaff", "#008000", "#fff300")

col_4  <- c("#00ff00", "#ff0000", "#0000ff", "#87cefa")
# qual_col_pals <- brewer.pal.info[brewer.pal.info$category == 'qual',]
# col_max <- unlist(mapply(brewer.pal, qual_col_pals$maxcolors, rownames(qual_col_pals)))


dim_reduc <- function(obj, atac = "atac", rna = "rna", adt = "adt") {
    DefaultAssay(obj) <- atac
    obj <-  RunTFIDF(obj) %>%
            FindTopFeatures(min.cutoff = "q25") %>%
            RunSVD(reduction.name = "lsi")

    DefaultAssay(obj) <- rna
    VariableFeatures(obj) <- rownames(obj)
    obj <-  NormalizeData(obj) %>%
            # FindVariableFeatures(nfeatures = 2000) %>%
            ScaleData() %>%
            RunPCA(reduction.name = "pca_rna", verbose = F)

    DefaultAssay(obj) <- adt
    VariableFeatures(obj) <- rownames(obj)
    obj <-  NormalizeData(obj, normalization.method = "CLR", margin = 2) %>%
            ScaleData() %>%
            RunPCA(reduction.name = "pca_adt", verbose = F)

    return(obj)
}


rename_task <- function(task) {
    for (data in c("dogma", "teadog")) {
        task <- gsub(paste0(data, "_full"       ), paste0(data, "-full"),
                gsub(paste0(data, "_paired_full"), paste0(data, "-paired+full"),
                gsub(paste0(data, "_paired_abc" ), paste0(data, "-paired-abc"),
                gsub(paste0(data, "_paired_ab"  ), paste0(data, "-paired-ab"),
                gsub(paste0(data, "_paired_ac"  ), paste0(data, "-paired-ac"),
                gsub(paste0(data, "_paired_bc"  ), paste0(data, "-paired-bc"),
                gsub(paste0(data, "_single_full"), paste0(data, "-diagonal+full"),
                gsub(paste0(data, "_single"     ), paste0(data, "-diagonal"),
                gsub(paste0(data, "_paired_a"   ), paste0(data, "-paired-a"),
                gsub(paste0(data, "_paired_b"   ), paste0(data, "-paired-b"),
                gsub(paste0(data, "_paired_c"   ), paste0(data, "-paired-c"),
                gsub(paste0(data, "_single_atac"), paste0(data, "-atac"),
                gsub(paste0(data, "_single_rna" ), paste0(data, "-rna"),
                gsub(paste0(data, "_single_adt" ), paste0(data, "-adt"), task))))))))))))))
    }
    task <- gsub("_transfer", " (model transfer)", task)
    return(task)
}

cm_plot <- function(pred, gt, keep_class = T, legend = T) {
    if (keep_class) {
        label_unique_gt <- str_sort(unique(c(gt, pred)))
        label_unique_pred <- label_unique_gt
    } else {
        #label_unique_gt <- str_sort(unique(gt))
        #label_unique_pred <- str_sort(unique(pred))
        label_unique_gt <- levels(gt)
        label_unique_pred <- levels(pred)
    }
    v1 <- character()
    v2 <- character()
    v3 <- numeric()
    for (lb1 in label_unique_pred) {
        for (lb2 in label_unique_gt) {
            v1 <- c(v1, lb1) 
            v2 <- c(v2, lb2) 
            if (sum(pred == lb1) == 0) {
                v3 <- c(v3, 0)
            } else {
                v3 <- c(v3,sum(which(gt == lb2) %in% which(pred == lb1)) / sum(gt == lb2))
            }
        }
    }
    v1 <- factor(v1, levels = label_unique_pred)
    v2 <- factor(v2, levels = label_unique_gt)
    cm <- data.frame(v1, v2, v3)

    plt <- ggplot(data = cm, aes(x = v1, y = v2, fill = v3)) + geom_tile() +
                  theme(axis.text.x = element_text(angle = 45, hjust = 1),
                        axis.text.y = element_text(angle = 45)) +
                #   scale_fill_continuous(name = "Consistency", limits = c(0, 1))
                  scale_fill_continuous(type = "viridis", name = "Consistency", limits = c(0, 1))
    
    if (!legend) {
        plt <- plt + NoLegend()
    }

    return(plt)
}

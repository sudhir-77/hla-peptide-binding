# Resume notes

## Draft bullets

- Trained a pan-allele peptide-MHC binding model (CNN) on 185985 IEDB measurements, reaching 0.939 binder AUROC and 0.750 Spearman vs. 0.880 for a per-allele PSSM baseline.
- Generalized to 10 alleles unseen in training, with mean per-allele AUROC 0.837 on the 9 alleles that contained both binders and non-binders.
- Built a neoantigen-ranking demo that scores every 9-mer of NY-ESO-1 (172 peptides) for HLA-A02:01.
- Compared a CNN and a small transformer on one fixed random split and one allele-held-out split, choosing the reported model by validation AUROC.

## Skills and tools

Python, PyTorch, NumPy, pandas, scikit-learn, matplotlib, pytest, BLOSUM62, CNN, transformer.

## Project summary

This project trains a pan-allele CNN and a small transformer to predict MHC class I binding from the peptide sequence and the MHC pseudo-sequence.
It compares both with a per-allele PSSM on a random split and on alleles left out of training, then ranks 9-mers from NY-ESO-1.

## GitHub description

Pan-allele MHC class I binding prediction on IEDB affinities, with a PSSM baseline and held-out alleles.

# PowerAnalysis
Matlab scripts for simulating power based upon pilot data. Currently works for the following study designs:
1) single-factor between-subjects designs (using independent samples t-tests)
2) single-factor within-subjects designs (using paired-samples t-tests)
3) two-factor within-subjects designs (using ANOVA and t-tests)
4) two-factor mixed-factor designs (using ANOVA and t-tests)

Still to come:
- two factor between-subjects designs
- analyses other than t-tests/ANOVA (e.g., mixed-effect modeling)
- data trimming, exclusion criteria, data transforms

Notes:
- currently need an equal number of trials for each condition for every pilot subject, or else will get an error
- please see PowerAnalysis_Guide.pdf for a guide to using the Matlab scripts.

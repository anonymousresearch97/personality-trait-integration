# Recheck saved primary CFA objects without fitting new models.
# Run after the six measurement fits; install lavaan in your normal R library.
suppressPackageStartupMessages(library(lavaan))
script_arg <- grep("^--file=", commandArgs(trailingOnly=FALSE), value=TRUE)
if (length(script_arg)) {
  script_path <- gsub("~+~", " ", sub("^--file=", "", script_arg[1]), fixed=TRUE)
  root <- dirname(dirname(normalizePath(script_path, mustWork=TRUE)))
} else {
  sourced <- Filter(Negate(is.null), lapply(sys.frames(), function(x) x$ofile))
  root <- if (length(sourced)) dirname(dirname(normalizePath(tail(sourced,1)[[1]], mustWork=TRUE))) else normalizePath('.')
}
source <- file.path(root, 'outputs', 'measurement_resolution')
out <- file.path(root, 'outputs', 'submission_code_audit', 'measurement')
dir.create(out, recursive=TRUE, showWarnings=FALSE)
metrics <- c('npar','chisq','df','cfi.robust','rmsea.robust','srmr')
records <- list()
for (domain in c('H','E','X','A','C','O')) {
  fits <- readRDS(file.path(source, paste0('primary_',domain,'_fits.rds')))
  summary <- read.csv(file.path(source,paste0('primary_',domain,'_fit_summary.csv')),check.names=FALSE)
  stopifnot(length(fits)==6, nrow(summary)==6)
  for(i in seq_len(nrow(summary))) {
    row <- summary[i,]
    fit <- fits[[paste(row$setting,row$age_group,sep=':')]]
    stopifnot(lavInspect(fit,'converged'),lavInspect(fit,'post.check'))
    observed <- as.numeric(fitMeasures(fit,metrics))
    expected <- as.numeric(row[,metrics])
    stopifnot(identical(is.na(observed),is.na(expected)),
              all(abs(observed[!is.na(observed)]-expected[!is.na(expected)]) < 1e-7))
    records[[length(records)+1]] <- data.frame(domain=domain,setting=row$setting,
      age_group=row$age_group,converged=TRUE,admissible=TRUE,
      max_summary_absolute_error=max(abs(observed-expected),na.rm=TRUE))
  }
  config <- fits[['multigroup_configural:all four']]
  metric <- fits[['multigroup_metric:all four']]
  stopifnot(fitMeasures(config,'df')==400,fitMeasures(metric,'df')==445,
            as.numeric(logLik(config))>=as.numeric(logLik(metric))-1e-5)
  expected_ns <- c('Adolescence 13-19'=7738L,'Early adulthood 20-39'=51670L,
                   'Middle adulthood 40-64'=35496L,'Older adulthood 65-100'=3915L)
  stopifnot(identical(as.integer(lavInspect(metric,'nobs')),
                     unname(expected_ns[lavInspect(metric,'group.label')])))
  pt <- parameterTable(metric)
  free <- pt[pt$op=='=~' & pt$free>0,]
  stopifnot(nrow(free)==60, sum(free$lhs=='D')==12, sum(pt$op=='==')==45)
  for(pair in unique(paste(free$lhs,free$rhs))) {
    rows <- free[paste(free$lhs,free$rhs)==pair,]
    stopifnot(nrow(rows)==4,length(unique(rows$label))==1,
              rows$label[1]!='',max(rows$est)-min(rows$est)<1e-9)
  }
  intercept <- pt[pt$op=='~1' & pt$lhs %in% lavNames(metric,'ov'),]
  stopifnot(nrow(intercept)==64,all(intercept$free>0),all(intercept$label==''))
}
write.csv(do.call(rbind,records),file.path(out,'saved_cfa_rechecks.csv'),row.names=FALSE)
capture.output(sessionInfo(),file=file.path(out,'R_session.txt'))
writeLines(c('PASS: 36 saved model objects, six metric constraints and group sample sizes independently rechecked.',
             'No fresh CFA fit was performed in this check. Prior independent named-facet H refits are archived separately.',
             'This verifies computation and constraints, not substantive measurement invariance.'),file.path(out,'audit_note.txt'))
cat('PASS: 36 CFA objects and six loading-constraint specifications; no new estimation.\n')

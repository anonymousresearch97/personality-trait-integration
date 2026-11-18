# Fixed domain-wise hierarchical configural/metric audit, robust ML.
suppressPackageStartupMessages(library(lavaan))
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=1 || !args[1] %in% c('H','E','X','A','C','O')) stop('Specify one domain: Rscript analysis/primary_measurement_resolution.R H (or E, X, A, C, O).')
domain <- args[1]
script_arg <- grep("^--file=", commandArgs(trailingOnly=FALSE), value=TRUE)
if (length(script_arg)) {
  script_path <- gsub("~+~", " ", sub("^--file=", "", script_arg[1]), fixed=TRUE)
  root <- dirname(dirname(normalizePath(script_path, mustWork=TRUE)))
} else {
  sourced <- Filter(Negate(is.null), lapply(sys.frames(), function(x) x$ofile))
  root <- if (length(sourced)) dirname(dirname(normalizePath(tail(sourced,1)[[1]], mustWork=TRUE))) else normalizePath('.')
}
out <- file.path(root,'outputs','measurement_resolution')
dir.create(out, recursive=TRUE, showWarnings=FALSE)
if (!file.exists(file.path(root,'data_prepared','primary_items.csv'))) stop('Missing data_prepared/primary_items.csv. Run python run_reproduction.py prepare-measurement from the bundle folder.')
dat <- read.csv(file.path(root,'data_prepared','primary_items.csv'),check.names=FALSE)
age_labels <- c('Adolescence 13-19','Early adulthood 20-39','Middle adulthood 40-64','Older adulthood 65-100')
dat$age_group <- factor(dat$age_group,levels=age_labels)
mapping <- read.csv(file.path(out,'primary_item_facet_map.csv'))
items <- mapping$column_name[mapping$domain==domain]
dat <- dat[,c(items,'age_group')]
model <- paste(readLines(file.path(out,paste0('primary_',domain,'_model.lav'))),collapse='\n')
warnings_record <- character()
fit_model <- function(data,group=NULL,equal=character()) {
  withCallingHandlers(cfa(model,data=data,group=group,group.equal=equal,
    estimator='MLR',meanstructure=TRUE,std.lv=FALSE,auto.cov.lv.x=FALSE,
    auto.cov.y=FALSE,se='none',test='yuan.bentler.mplus',
    control=list(iter.max=3000)),warning=function(w){
      warnings_record <<- c(warnings_record,conditionMessage(w));invokeRestart('muffleWarning')
    })
}
rows <- list(); fits <- list()
record_fit <- function(fit,setting,age='all four') {
  fm <- fitMeasures(fit,c('npar','chisq','df','pvalue','cfi','tli','rmsea','srmr',
       'chisq.scaled','pvalue.scaled','cfi.robust','tli.robust','rmsea.robust',
       'rmsea.ci.lower.robust','rmsea.ci.upper.robust'))
  pt <- parameterEstimates(fit)
  variances <- pt[pt$op=='~~' & pt$lhs==pt$rhs,]
  lv_cov <- lavInspect(fit,'cov.lv')
  if(!is.list(lv_cov))lv_cov<-list(lv_cov)
  min_eigen <- min(vapply(lv_cov,function(x)min(eigen(x,symmetric=TRUE,only.values=TRUE)$values),0.0))
  row <- data.frame(domain=domain,setting=setting,age_group=age,
       converged=lavInspect(fit,'converged'),minimum_variance=min(variances$est),
       minimum_latent_eigenvalue=min_eigen,as.list(fm),check.names=FALSE)
  row$admissible <- row$converged & row$minimum_variance>=0 & min_eigen>0
  rows[[length(rows)+1]] <<- row
  fits[[paste(setting,age,sep=':')]] <<- fit
  write.csv(pt,file.path(out,paste0('primary_',domain,'_',setting,'_',gsub('[^a-zA-Z0-9]','_',age),'_parameters.csv')),row.names=FALSE)
  write.csv(do.call(rbind,rows),file.path(out,paste0('primary_',domain,'_fit_summary.csv')),row.names=FALSE)
  cat(domain,setting,age,'CFIrobust',fm['cfi.robust'],'RMSEArobust',fm['rmsea.robust'],'SRMR',fm['srmr'],'admissible',row$admissible,'\n');flush.console()
}
for(age in age_labels){
  fit <- fit_model(dat[dat$age_group==age,items,drop=FALSE])
  record_fit(fit,'single_configural',age)
}
config <- fit_model(dat,group='age_group')
record_fit(config,'multigroup_configural')
metric <- fit_model(dat,group='age_group',equal='loadings')
record_fit(metric,'multigroup_metric')
stopifnot(lavInspect(config,'converged'),lavInspect(metric,'converged'))
stopifnot(fitMeasures(metric,'df')-fitMeasures(config,'df')==45)
stopifnot(as.numeric(logLik(config))>=as.numeric(logLik(metric))-1e-5)
# Confirm loadings constrained on BOTH levels and nuisance intercepts free.
pt <- parameterTable(metric)
free_load <- pt[pt$op=='=~' & pt$free>0,]
for(pair in unique(paste(free_load$lhs,free_load$rhs))){
  sub<-free_load[paste(free_load$lhs,free_load$rhs)==pair,]
  stopifnot(nrow(sub)==4,length(unique(sub$label))==1,sub$label[1]!='')
}
write.csv(data.frame(domain=domain,lavaan_version=as.character(packageVersion('lavaan')),
  config_df=fitMeasures(config,'df'),metric_df=fitMeasures(metric,'df'),
  config_loglik=as.numeric(logLik(config)),metric_loglik=as.numeric(logLik(metric)),
  metric_free_loading_rows=nrow(free_load),metric_distinct_free_loading_labels=length(unique(free_load$label)),
  delta_cfi_robust=fitMeasures(metric,'cfi.robust')-fitMeasures(config,'cfi.robust'),
  delta_rmsea_robust=fitMeasures(metric,'rmsea.robust')-fitMeasures(config,'rmsea.robust'),
  delta_srmr=fitMeasures(metric,'srmr')-fitMeasures(config,'srmr')),
  file.path(out,paste0('primary_',domain,'_metric_comparison.csv')),row.names=FALSE)
capture.output(lavTestLRT(config,metric),file=file.path(out,paste0('primary_',domain,'_robust_difference_test.txt')))
saveRDS(fits,file.path(out,paste0('primary_',domain,'_fits.rds')))
writeLines(unique(warnings_record),file.path(out,paste0('primary_',domain,'_warnings.txt')))
capture.output(sessionInfo(),file=file.path(out,paste0('primary_',domain,'_session_info.txt')))

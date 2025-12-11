# ==============================================================================
# ANALISI DEI VINI CON CHAID, CART, BAGGING E RANDOM FOREST
# ==============================================================================

# PACCHETTI
library(dplyr)
library(CHAID)
library(partykit)
library(caret)  
library(rpart)          
library(rpart.plot)     
library(randomForest) 

# Caricamento dati
wines <- read.table("wines2.txt", header = TRUE, sep = "\t", dec = ".", stringsAsFactors = FALSE)

# Ricodifica Variabile Risposta (Target)
wines$Class3 <- cut(wines$Classificazione,
                    breaks = c(0, 2, 3, 5),
                    labels = c("(0,2]", "(2,3]", "(3,5]"),
                    right = TRUE, include.lowest = TRUE)

# Fattorizzazione variabile target
wines$Class3 <- factor(wines$Class3, ordered = TRUE)

# SOLO Variabili Quantitative
quant_vars <- c("Alcohol", "Malic.acid", "Ash", "Alcalinity.of.ash",
                "Magnesium", "Total.phenols", "Flavanoids", 
                "Nonflavanoid.phenols", "Proanthocyanins", 
                "Color.intensity", "Hue", 
                "OD280.OD315.of.diluted.wines", "Proline")

# Parametri CHAID
ctrl <- chaid_control(minsplit = 20, minbucket = 7, alpha2 = 0.05)

# ------------------------------------------------------------------------------
# 2. DEFINIZIONE FUNZIONI HELPER
# ------------------------------------------------------------------------------

# Funzione Output Performance
print_performance <- function(model_obj, data_obj, target_col, title="") {
  preds <- predict(model_obj, newdata = data_obj)
  actual <- data_obj[[target_col]]
  
  tbl <- table(Osservato = actual, Previsto = preds)
  acc <- sum(diag(tbl)) / sum(tbl)
  err <- 1 - acc
  
  cat(paste0("\n--- PERFORMANCE: ", title, " ---\n"))
  cat(sprintf("Accuratezza: %.2f%% | Errore: %.2f%%\n", acc*100, err*100))
  print(tbl)
  return(err)
}

# --- Metodi di Split ---

find_split_chisq <- function(x, y) {
  ux <- sort(unique(x)); if(length(ux) < 2) return(ux[1])
  stats <- sapply(ux, function(t) {
    tryCatch(suppressWarnings(chisq.test(table(x < t, y))$statistic), error=function(e) 0)
  })
  ux[which.max(stats)]
}

find_split_gini <- function(x, y) {
  ux <- sort(unique(x)); if(length(ux) < 2) return(ux[1])
  ginis <- sapply(ux, function(t) {
    l <- x < t; r <- !l
    if(sum(l)==0 | sum(r)==0) return(1)
    g_l <- 1 - sum((table(y[l])/sum(l))^2)
    g_r <- 1 - sum((table(y[r])/sum(r))^2)
    (sum(l)/length(x))*g_l + (sum(r)/length(x))*g_r
  })
  ux[which.min(ginis)]
}

find_split_entropy <- function(x, y) {
  entropy <- function(k) { p <- k/sum(k); -sum(p * log2(p + 1e-10)) }
  base_ent <- entropy(table(y))
  ux <- sort(unique(x)); if(length(ux) < 2) return(ux[1])
  gains <- sapply(ux, function(t) {
    l <- x < t; r <- !l
    if(sum(l)==0 | sum(r)==0) return(0)
    ent_after <- (sum(l)/length(x))*entropy(table(y[l])) + (sum(r)/length(x))*entropy(table(y[r]))
    base_ent - ent_after
  })
  ux[which.max(gains)]
}

apply_binary <- function(col, thr) {
  if(is.na(thr)) return(factor(rep(NA, length(col))))
  factor(ifelse(col < thr, "Low", "High"), levels=c("Low","High"))
}

get_robust_breaks <- function(vec) {
  brks <- unique(quantile(vec, probs=seq(0,1,0.25), na.rm=TRUE))
  if(length(brks) < 3) brks <- unique(quantile(vec, probs=c(0,0.5,1), na.rm=TRUE))
  brks[1] <- -Inf
  brks[length(brks)] <- Inf
  return(brks)
}

# ==============================================================================
# PARTE A: METODO BINARIO (CHI-SQUARE)
# ==============================================================================

# A1. CAMPIONE TOTALE
w_bin <- wines %>% select(Class3, all_of(quant_vars))
for(v in quant_vars) {
  thr <- find_split_chisq(wines[[v]], wines$Class3)
  w_bin[[v]] <- apply_binary(wines[[v]], thr)
}

fit_bin <- chaid(Class3 ~ ., data = w_bin, control = ctrl)
plot(fit_bin, main="Metodo Binario - Campione Totale (Solo Chimiche)")
err_bin_full <- print_performance(fit_bin, w_bin, "Class3", "BINARIO - FULL DATA")

# A2. TRAIN/TEST SPLIT (70/30)
set.seed(123)
idx <- createDataPartition(wines$Class3, p=0.7, list=FALSE)
d_tr <- wines[idx,]; d_te <- wines[-idx,]

tr_bin <- d_tr %>% select(Class3, all_of(quant_vars))
te_bin <- d_te %>% select(Class3, all_of(quant_vars))

for(v in quant_vars) {
  thr <- find_split_chisq(d_tr[[v]], d_tr$Class3)
  tr_bin[[v]] <- apply_binary(d_tr[[v]], thr)
  te_bin[[v]] <- apply_binary(d_te[[v]], thr)
}

fit_bin_split <- chaid(Class3 ~ ., data = tr_bin, control = ctrl)
plot(fit_bin_split, main="Metodo Binario - Train/Test Split (Solo Chimiche)")
err_bin_split <- print_performance(fit_bin_split, te_bin, "Class3", "BINARIO - TEST SET")

# A3. CROSS VALIDATION
set.seed(999)
folds <- createFolds(wines$Class3, k=5)
cv_errs_bin <- numeric(5)

for(k in 1:5) {
  idx_k <- folds[[k]]
  d_tr <- wines[-idx_k,]; d_te <- wines[idx_k,]
  tr_b <- d_tr %>% select(Class3, all_of(quant_vars))
  te_b <- d_te %>% select(Class3, all_of(quant_vars))
  
  for(v in quant_vars) {
    thr <- find_split_chisq(d_tr[[v]], d_tr$Class3)
    tr_b[[v]] <- apply_binary(d_tr[[v]], thr)
    te_b[[v]] <- apply_binary(d_te[[v]], thr)
  }
  
  try({
    ft <- chaid(Class3 ~ ., data=tr_b, control=ctrl)
    p <- predict(ft, newdata=te_b)
    cv_errs_bin[k] <- 1 - mean(p == te_b$Class3)
  }, silent=TRUE)
}
cat(sprintf("Errore Medio CV (Binario): %.2f%%\n", mean(cv_errs_bin)*100))

# ==============================================================================
# PARTE B: METODO QUARTILI
# ==============================================================================

# B1. CAMPIONE TOTALE
w_qt <- wines %>% select(Class3, all_of(quant_vars))
for(v in quant_vars) {
  brks <- get_robust_breaks(wines[[v]])
  lbls <- paste0("Q", 1:(length(brks)-1))
  w_qt[[v]] <- cut(wines[[v]], breaks=brks, labels=lbls, include.lowest=TRUE, ordered_result=TRUE)
}

fit_qt <- chaid(Class3 ~ ., data = w_qt, control = ctrl)
plot(fit_qt, main="Metodo Quartili - Campione Totale (Solo Chimiche)")
err_qt_full <- print_performance(fit_qt, w_qt, "Class3", "QUARTILI - FULL DATA")

# B2. TRAIN/TEST SPLIT
set.seed(123)
idx <- createDataPartition(wines$Class3, p=0.7, list=FALSE)
d_tr <- wines[idx,]; d_te <- wines[-idx,]

tr_qt <- d_tr %>% select(Class3, all_of(quant_vars))
te_qt <- d_te %>% select(Class3, all_of(quant_vars))

for(v in quant_vars) {
  my_breaks <- get_robust_breaks(d_tr[[v]])
  my_labels <- paste0("Q", 1:(length(my_breaks)-1))
  tr_qt[[v]] <- cut(d_tr[[v]], breaks=my_breaks, labels=my_labels, include.lowest=TRUE, ordered_result=TRUE)
  te_qt[[v]] <- cut(d_te[[v]], breaks=my_breaks, labels=my_labels, include.lowest=TRUE, ordered_result=TRUE)
}

fit_qt_split <- chaid(Class3 ~ ., data = tr_qt, control = ctrl)
plot(fit_qt_split, main="Metodo Quartili - Train/Test Split (Solo Chimiche)")
err_qt_split <- print_performance(fit_qt_split, te_qt, "Class3", "QUARTILI - TEST SET")

# B3. CROSS VALIDATION
set.seed(999)
folds <- createFolds(wines$Class3, k=5)
cv_errs_qt <- numeric(5)
cv_models_qt <- list() 

for(k in 1:5) {
  idx_k <- folds[[k]]
  d_tr <- wines[-idx_k,]; d_te <- wines[idx_k,]
  tr_q <- d_tr %>% select(Class3, all_of(quant_vars))
  te_q <- d_te %>% select(Class3, all_of(quant_vars))
  
  for(v in quant_vars) {
    my_breaks <- get_robust_breaks(d_tr[[v]])
    my_labels <- paste0("Q", 1:(length(my_breaks)-1))
    tr_q[[v]] <- cut(d_tr[[v]], breaks=my_breaks, labels=my_labels, include.lowest=TRUE, ordered_result=TRUE)
    te_q[[v]] <- cut(d_te[[v]], breaks=my_breaks, labels=my_labels, include.lowest=TRUE, ordered_result=TRUE)
  }
  
  ft <- chaid(Class3 ~ ., data=tr_q, control=ctrl)
  cv_models_qt[[k]] <- ft
  p <- predict(ft, newdata=te_q)
  cv_errs_qt[k] <- 1 - mean(p == te_q$Class3, na.rm=TRUE)
  cat(sprintf("   Fold %d -> Errore: %.2f%%\n", k, cv_errs_qt[k]*100))
}

mean_err <- mean(cv_errs_qt)
cat(sprintf("\nErrore Medio CV (Quartili): %.2f%%\n", mean_err*100))

best_fold_idx <- which.min(cv_errs_qt)
best_cv_tree <- cv_models_qt[[best_fold_idx]]
plot(best_cv_tree, 
     main = paste0("Miglior Albero CV (Quartili) - Fold ", best_fold_idx, 
                   "\n(Err. Test Fold: ", round(cv_errs_qt[best_fold_idx]*100, 2), "%)"),
     gp = gpar(fontsize = 8))

# ==============================================================================
# PARTE C: METODI AVANZATI (GINI, ENTROPIA, INTERVAL)
# ==============================================================================

err_bin_mean <- mean(cv_errs_bin)
err_qt_mean  <- mean(cv_errs_qt)

cv_results_list <- data.frame(
  Metodo = c("Binario(Chisq)", "Quartili"),
  Error_Mean = c(err_bin_mean, err_qt_mean),
  Error_Min_Fold = c(min(cv_errs_bin), min(cv_errs_qt)),
  Error_Max_Fold = c(max(cv_errs_bin), max(cv_errs_qt))
)

adv_methods <- c("Gini", "Entropy", "Interval")

for(method in adv_methods) {
  cat(sprintf("\n--- Elaborazione Metodo: %s ---\n", toupper(method)))
  current_cv_errs <- numeric(5)
  
  for(k in 1:5) {
    idx_k <- folds[[k]]
    d_tr <- wines[-idx_k,]; d_te <- wines[idx_k,]
    tr_adv <- d_tr %>% select(Class3, all_of(quant_vars))
    te_adv <- d_te %>% select(Class3, all_of(quant_vars))
    
    for(v in quant_vars) {
      if(method == "Gini") {
        thr <- find_split_gini(d_tr[[v]], d_tr$Class3)
        tr_adv[[v]] <- apply_binary(d_tr[[v]], thr)
        te_adv[[v]] <- apply_binary(d_te[[v]], thr)
        
      } else if(method == "Entropy") {
        thr <- find_split_entropy(d_tr[[v]], d_tr$Class3)
        tr_adv[[v]] <- apply_binary(d_tr[[v]], thr)
        te_adv[[v]] <- apply_binary(d_te[[v]], thr)
        
      } else if(method == "Interval") {
        rng <- range(d_tr[[v]], na.rm = TRUE)
        cuts <- seq(from = rng[1], to = rng[2], length.out = 4)
        cuts[1] <- -Inf; cuts[length(cuts)] <- Inf
        tr_adv[[v]] <- cut(d_tr[[v]], breaks=cuts, include.lowest=T, ordered_result=T)
        te_adv[[v]] <- cut(d_te[[v]], breaks=cuts, include.lowest=T, ordered_result=T)
      }
    }
    
    tryCatch({
      m <- chaid(Class3 ~ ., data=tr_adv, control=ctrl)
      p <- predict(m, newdata=te_adv)
      current_cv_errs[k] <- 1 - mean(p == te_adv$Class3, na.rm=TRUE)
    }, error=function(e) current_cv_errs[k] <<- NA)
  }
  
  mean_err <- mean(current_cv_errs, na.rm=TRUE)
  cat(sprintf(">> Errore Medio CV (%s): %.2f%%\n", method, mean_err*100))
  
  cv_results_list <- rbind(cv_results_list, data.frame(
    Metodo = method, 
    Error_Mean = mean_err,
    Error_Min_Fold = min(current_cv_errs, na.rm=T),
    Error_Max_Fold = max(current_cv_errs, na.rm=T)
  ))
}

# ==============================================================================
# PARTE D: SELEZIONE MIGLIOR MODELLO
# ==============================================================================

cv_results_list <- na.omit(cv_results_list)
final_ranking <- cv_results_list %>% arrange(Error_Mean)
print(final_ranking)

best_method <- final_ranking$Metodo[1]
best_err    <- final_ranking$Error_Mean[1]

cat(sprintf(" IL METODO MIGLIORE È: %s\n", toupper(best_method)))
cat(sprintf("   Errore MEDIO più basso in CV: %.2f%%\n", best_err*100))

# RICOSTRUZIONE MODELLO FINALE
final_df <- wines %>% select(Class3, all_of(quant_vars))
cat(">> Ricostruzione dataset con metodo migliore...\n")

for(v in quant_vars) {
  vals <- wines[[v]]
  y <- wines$Class3
  
  if(grepl("Binario", best_method)) {
    thr <- find_split_chisq(vals, y)
    final_df[[v]] <- apply_binary(vals, thr)
  } else if(best_method == "Gini") {
    thr <- find_split_gini(vals, y)
    final_df[[v]] <- apply_binary(vals, thr)
  } else if(best_method == "Entropy") {
    thr <- find_split_entropy(vals, y)
    final_df[[v]] <- apply_binary(vals, thr)
  } else if(best_method == "Quartili") {
    brks <- get_robust_breaks(vals) 
    final_df[[v]] <- cut(vals, breaks=brks, labels=paste0("Q",1:(length(brks)-1)), include.lowest=T, ordered_result=T)
  } else if(best_method == "Interval") {
    rng <- range(vals, na.rm = TRUE)
    cuts <- seq(from = rng[1], to = rng[2], length.out = 4)
    cuts[1] <- -Inf; cuts[length(cuts)] <- Inf
    final_df[[v]] <- cut(vals, breaks=cuts, include.lowest=T, ordered_result=T)
  }
}

final_tree <- chaid(Class3 ~ ., data = final_df, control = ctrl)
plot(final_tree, main = paste("ALBERO FINALE:", toupper(best_method), 
                              "\n(Solo variabili quantitative - Full Data)"))
print_performance(final_tree, final_df, "Class3", "MODELLO MIGLIORE (FULL DATA)")
table(final_tree$fitted)

# Test Chi-Quadro
cat("\n>> TEST CHI-QUADRO (Target vs Variabili)\n")
chi_stats <- data.frame(Variabile=character(), P_Value=numeric(), stringsAsFactors=F)
for(pred in setdiff(names(final_df), "Class3")) {
  tst <- suppressWarnings(chisq.test(table(final_df$Class3, final_df[[pred]])))
  chi_stats <- rbind(chi_stats, data.frame(Variabile=pred, P_Value=tst$p.value))
}
chi_stats$Significativo <- ifelse(chi_stats$P_Value < 0.05, "***", "")
chi_stats <- chi_stats %>% arrange(P_Value)
print(chi_stats)

# ==============================================================================
# PARTE E: TEST SU NUOVI VINI (SOLO CHIMICHE)
# ==============================================================================

new_wines_data <- data.frame(
  ID = 1:5,
  Classificazione_Originale = c(1, 5, 1, 1, 4), 
  Alcohol = c(14.5, 12.0, 14.0, 15.5, 11.5),
  Malic.acid = c(1.8, 3.5, 2.0, 1.5, 2.5),
  Ash = c(2.4, 2.2, 2.3, 2.5, 2.1),
  Alcalinity.of.ash = c(15, 20, 18, 16, 21),
  Magnesium = c(110, 85, 95, 120, 88),
  Total.phenols = c(3.0, 1.5, 2.2, 3.5, 1.8),
  Flavanoids = c(3.2, 0.8, 2.0, 3.9, 0.9),
  Nonflavanoid.phenols = c(0.2, 0.5, 0.3, 0.2, 0.4),
  Proanthocyanins = c(1.8, 0.9, 1.5, 2.0, 1.1),
  Color.intensity = c(5.5, 3.0, 4.5, 6.0, 3.5),
  Hue = c(1.1, 0.7, 0.95, 1.2, 0.8),
  OD280.OD315.of.diluted.wines = c(3.5, 1.5, 1.45, 3.8, 1.6),
  Proline = c(1200, 400, 1100, 1600, 450)
)

new_wines_data$Target_Atteso <- cut(new_wines_data$Classificazione_Originale,
                                    breaks = c(0, 2, 3, 5),
                                    labels = c("(0,2]", "(2,3]", "(3,5]"),
                                    right = TRUE, include.lowest = TRUE)

new_wines_ready <- new_wines_data %>% select(all_of(c("ID", "Classificazione_Originale", 
                                                      "OD280.OD315.of.diluted.wines", 
                                                      quant_vars)))

for(v in quant_vars) {
  rng_train <- range(wines[[v]], na.rm = TRUE)
  cuts <- seq(from = rng_train[1], to = rng_train[2], length.out = 4)
  cuts[1] <- -Inf; cuts[length(cuts)] <- Inf
  new_wines_ready[[v]] <- cut(new_wines_data[[v]], breaks = cuts, 
                              include.lowest = TRUE, ordered_result = TRUE)
}

new_wines_ready$Target_Atteso <- new_wines_data$Target_Atteso
new_wines_ready$OD280 <- new_wines_data$OD280.OD315.of.diluted.wines

predictions <- predict(final_tree,
                       newdata = new_wines_ready %>% select(all_of(quant_vars)))

new_wines_ready$PREDIZIONE <- predictions

confronto <- data.frame(
  ID = new_wines_ready$ID,
  Classe_Reale_Num = new_wines_ready$Classificazione_Originale,
  OD280 = new_wines_ready$OD280,
  Target_Reale_Grp = new_wines_ready$Target_Atteso,                
  PREDIZIONE = new_wines_ready$PREDIZIONE,                                  
  ESITO = ifelse(as.character(new_wines_ready$Target_Atteso) == 
                   as.character(new_wines_ready$PREDIZIONE), "CORRETTO", "ERRORE")
)

cat("\n--- VERIFICA CORRISPONDENZA CLASSI (SOLO CHIMICHE) ---\n")
print(confronto)
acc_test <- mean(confronto$ESITO == "CORRETTO")
cat(sprintf("\nAccuratezza sul dataset fittizio: %.0f%%\n", acc_test * 100))



# ==============================================================================
# PARTE H: CART, BAGGING E RANDOM FOREST
# ==============================================================================

data_ml <- wines %>% select(Class3, all_of(quant_vars))

# Divisione Train (70%) e Test (30%)
set.seed(123)
train_index <- sample(1:nrow(data_ml), round(0.7 * nrow(data_ml)))
d_train <- data_ml[train_index, ]
d_test  <- data_ml[-train_index, ]

# ==============================================================================
# H1 CART (ALBERO SINGOLO) CON PRUNING
# ==============================================================================

# A. Albero di taglia massima
tree_full <- rpart(Class3 ~ ., data = d_train, method = "class",
                   control = rpart.control(cp = 0.0001, minsplit = 10))

# B. Tabella di complessità (CP Table)
print(tree_full$cptable)

# Grafico dell'errore relativo
plotcp(tree_full) 


# C. CP ottimale (quello con l'errore minore)
opt_index <- which.min(tree_full$cptable[, "xerror"])
opt_cp    <- tree_full$cptable[opt_index, "CP"]
cat(sprintf("CP Ottimale trovato: %f\n", opt_cp))

# D. Potatura  dell'albero (Pruning)
tree_pruned <- prune(tree_full, cp = opt_cp)
summary(tree_pruned)

# E. Grafico dell'Albero Finale
rpart.plot(tree_pruned, type = 4, extra = 104, 
           main = "Albero CART Ottimizzato (Pruned)")

# F. Performance sul Test Set
pred_cart <- predict(tree_pruned, newdata = d_test, type = "class")
acc_cart  <- mean(as.character(pred_cart) == as.character(d_test$Class3))
cat(sprintf("Accuratezza CART (Test Set): %.2f%%\n", acc_cart * 100))


# ==============================================================================
# H2 BAGGING (Bootstrap Aggregating)
# ==============================================================================

set.seed(123)
bag_mod <- randomForest(Class3 ~ ., data = d_train, 
                        mtry = length(quant_vars), 
                        ntree = 500, 
                        importance = TRUE)

# A. Errore Out-Of-Bag (OOB)
oob_err_bag <- bag_mod$err.rate[500, "OOB"]

# B. Grafico Importanza Variabili
varImpPlot(bag_mod, main = "BAGGING: Importanza Variabili")

# C. Performance sul Test Set
pred_bag <- predict(bag_mod, newdata = d_test)
acc_bag  <- mean(as.character(pred_bag) == as.character(d_test$Class3))

cat(sprintf("Accuratezza Bagging (Test Set): %.2f%%\n", acc_bag * 100))
cat(sprintf("Errore Stimato OOB:             %.2f%%\n", oob_err_bag * 100))


# ==============================================================================
# H3 RANDOM FOREST (RF)
# ==============================================================================

set.seed(123)
rf_mod <- randomForest(Class3 ~ ., data = d_train, 
                       mtry = floor(sqrt(length(quant_vars))), 
                       ntree = 500, 
                       importance = TRUE)

# A. Errore Out-Of-Bag (OOB)
oob_err_rf <- rf_mod$err.rate[500, "OOB"]

# B. Grafico Importanza Variabili
varImpPlot(rf_mod, main = "RANDOM FOREST: Importanza Variabili")

# C. Performance sul Test Set
pred_rf <- predict(rf_mod, newdata = d_test)
acc_rf  <- mean(as.character(pred_rf) == as.character(d_test$Class3))

cat(sprintf("Accuratezza RF (Test Set):      %.2f%%\n", acc_rf * 100))
cat(sprintf("Errore Stimato OOB:             %.2f%%\n", oob_err_rf * 100))


# ==============================================================================
# H4 TABELLA RIEPILOGATIVA FINALE
# ==============================================================================

results <- data.frame(
  Modello = c("CART (Pruned)", "Bagging", "Random Forest"),
  Accuratezza_Test = c(acc_cart, acc_bag, acc_rf) * 100,
  Errore_OOB_Stimato = c(NA, oob_err_bag * 100, oob_err_rf * 100)
)

# Classifica dal migliore al peggiore (in base all'accuratezza sul test set)
results <- results[order(results$Accuratezza_Test, decreasing = TRUE), ]
print(results)


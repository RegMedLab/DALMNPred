library(multiMiR)

t=read.table("mirna_list.txt")
t= as.data.frame(t)
list_mirna= t$V1


example <- get_multimir(org = "hsa",
                        mirna = list_mirna,
                        predicted.cutoff = 20,
                        predicted.cutoff.type = "p",
                        table   = "predicted",
                        summary = TRUE)

example_val <- get_multimir(org = "hsa",
                            mirna= list_mirna,
                            table   = "validated",
                            predicted.cutoff = 20,
                            predicted.cutoff.type = "p",
                            summary = TRUE)

write.csv(example@data, file = "combined_predicted_common_mirnaexp_lncbase.csv")
write.csv(example_val@data, file = "combined_validated_common_mirnaexp_lncbase.csv")
View(example_val@data)

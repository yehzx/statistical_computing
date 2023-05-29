install.packages("LearnBayes")
library(LearnBayes)
data("birthweight")
write.csv(birthweight, "./birthweight.csv", row.names=FALSE)

library(MASS)
library(openxlsx)
set.seed(2024)


J = 10
q = 3
K = 2
L = 3

w <- matrix(c(0.5, 0.5, 0
                0, 0.5 0.5), nrow = 5, byrow=T)
cat_w <- 1:L 

pi <- c(0.5, 0.5)
cat_pi <- 1:K

nj <- c(3:5, 3:5, 3:5, 3:5, 3:5)

mu1 <- numeric(3)
mu2 <- c(9:11)
mu3 <- -c(9:11)

sigma1 <- diag(1, nrow = q, ncol = q)
sigma2 <- matrix(c(16, 4, 4, 4, 9, 2, 4, 2, 9), nrow = 3, byrow = TRUE)

nrow_data <- sum(nj)
temp <- numeric(nrow_data)
data <- data.frame(Subject = temp, c1 = temp, c2 = temp, c3 = temp, DC = temp, OC = temp)
pos = 0

for (j in 1:J){
  s1 <- sample(cat_pi, size = 1, prob = pi)
  for(i in 1:nj[j]){
    pos <- pos+1
    s2 <- sample(cat_w, size = 1, prob = w[s1,])
    data$DC[pos] <- s1
    data$OC[pos] <- s2
    data$Subject[pos] <- j
    if (s2 == 1)
      c <- mvrnorm(n = 1, mu = mu1, Sigma = sigma1)
    if (s2 == 2)
      c <- mvrnorm(n = 1, mu = mu1, Sigma = sigma2)
    if (s2 == 3)
      c <- mvrnorm(n = 1, mu = mu2, Sigma = sigma1)
    if (s2 == 4)
      c <- mvrnorm(n = 1, mu = mu2, Sigma = sigma2)
    data$c1[pos] <- c[1]
    data$c2[pos] <- c[2]
    data$c3[pos] <- c[3]
  }
}

#save as an excell
write.csv(data, file = "test_data.csv", row.names = FALSE)


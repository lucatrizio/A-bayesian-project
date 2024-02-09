mixture_dags=function(mu_mat,Sigma_mat,A_mat,L,q,n){
  #converto in array tridimensionali
  
  sink("output.txt")
  
  # Print the value of the variable
  print(mu_mat)
  print(Sigma_mat)
  print(A_mat)
  
  # Stop redirecting output
  sink()

  a_pi=1
  b_pi=(2*q-2)/3
  
  U    = diag(1,q)
  a    = q
  m_0  = rep(0, q)
  a_mu = 1
  
  A_chain =  array(0, c(q, q, L))   #L-> numb of observational cluster
  Sigma_chain = array(0, c(q, q, L)) 
  mu_chain= array(0, c(1, q, L))
  
  for (k in 1:L) {
    mu_chain[,,k] <- mu_mat[[k]]
  }
  
  for (k in 1:L) {
    Sigma_chain[,,k] <- Sigma_mat[[k]]
  }
  
  for (k in 1:L) {
    A_chain[,,k] <- A_mat[[k]]
  }
  
  A_chain_new =  array(0, c(q, q, L))   #L-> numb of observational cluster
  Sigma_chain_new = array(0, c(q, q, L)) 
  mu_chain_new= array(0, c(1, q, L))
  
  for(k in 1:L){
    
     DAG =A_chain[,,k]
     move_star = move(A = DAG, q = q)
     
     DAG_star   = move_star$A_new
     type.op    = move_star$type.operator
     nodes_star = move_star$nodes
     
     X_k=Sigma_chain[,,k]
     #n_k =numero di osservazioni che cadono nel cluster osservazionale i=1,...,L
     n_k=n[k]
     x_bar  = colMeans(X_k)
     X_zero = t((t(X_k) - x_bar))
     
     # Posterior hyper-parameters for DAG k
     a_tilde    = a + n_k
     U_tilde    = U + t(X_zero)%*%X_zero + (a_mu*n_k)/(a_mu + n_k)*(x_bar - m_0)%*%t(x_bar - m_0)
     a_mu_tilde = a_mu + n_k
     m_0_tilde  = a_mu/(a_mu + n_k)*m_0 + n_k/(a_mu + n_k)*x_bar
     
     # Multiplicity correction (log)prior
     #edge=numero di 1, sum(DAG_star)
     logprior.new = lgamma(n.edge(DAG_star) + a_pi) + 
       lgamma(q*(q-1)/2 - n.edge(DAG_star) + b_pi - 1)
     
     logprior.old = lgamma(n.edge(DAG) + a_pi) + 
       lgamma(q*(q-1)/2 - n.edge(DAG) + b_pi - 1)
     
     logprior = logprior.new - logprior.old
     
     # Distinguish 3 cases:
     
     if(type.op == 1){
       
       # (1) Insert a directed edge
       
       marg_star = norm_const_j(j = nodes_star[2], A = DAG_star, U = U, a = a, m = m_0, a_mu = a_mu) -
         norm_const_j(j = nodes_star[2], A = DAG_star, U = U_tilde, a = a_tilde, m = m_0_tilde, a_mu = a_mu_tilde)
       
       marg      = norm_const_j(j = nodes_star[2], A = DAG, U = U, a = a, m = m_0, a_mu = a_mu) -
         norm_const_j(j = nodes_star[2], A = DAG, U = U_tilde, a = a_tilde, m = m_0_tilde, a_mu = a_mu_tilde)
       
     }else{
       
       if(type.op == 2){
         
         # (2) Delete a directed edge
         
         marg_star = norm_const_j(j = nodes_star[2], A = DAG_star, U = U, a = a, m = m_0, a_mu = a_mu) -
           norm_const_j(j = nodes_star[2], A = DAG_star, U = U_tilde, a = a_tilde, m = m_0_tilde, a_mu = a_mu_tilde)
         
         marg      = norm_const_j(j = nodes_star[2], A = DAG, U = U, a = a, m = m_0, a_mu = a_mu) -
           norm_const_j(j = nodes_star[2], A = DAG, U = U_tilde, a = a_tilde, m = m_0_tilde, a_mu = a_mu_tilde)
         
       }else{
         
         # (3) Reverse a directed edge
         
         marg_star = norm_const_j(j = nodes_star[1], A = DAG_star, U = U, a = a, m = m_0, a_mu = a_mu) -
           norm_const_j(j = nodes_star[1], A = DAG_star, U = U_tilde, a = a_tilde, m = m_0_tilde, a_mu = a_mu_tilde) +
           norm_const_j(j = nodes_star[2], A = DAG_star, U = U, a = a, m = m_0, a_mu = a_mu) -
           norm_const_j(j = nodes_star[2], A = DAG_star, U = U_tilde, a = a_tilde, m = m_0_tilde, a_mu = a_mu_tilde)
         
         marg      = norm_const_j(j = nodes_star[1], A = DAG, U = U, a = a, m = m_0, a_mu = a_mu) -
           norm_const_j(j = nodes_star[1], A = DAG, U = U_tilde, a = a_tilde, m = m_0_tilde, a_mu = a_mu_tilde) +
           norm_const_j(j = nodes_star[2], A = DAG, U = U, a = a, m = m_0, a_mu = a_mu) -
           norm_const_j(j = nodes_star[2], A = DAG, U = U_tilde, a = a_tilde, m = m_0_tilde, a_mu = a_mu_tilde)
         
       }
       
     }
     
     # acceptance ratio
     
     ratio_D = min(0, marg_star - marg + logprior)
     
     # accept DAG
     
     if(log(runif(1)) < ratio_D){
       
       A_chain_new[,,k] = DAG_star #Ricordo k=observational cluster k
       #aggiorno matrice di adiacenza 
     }
     
     else if (log(runif(1)) >= ratio_D){
       A_chain_new[,,k]=A_chain[,,k]
     }
     ############################################################################
     ## Sample from the posterior of Sigma_k and mu_k conditionally on DAG D_k ##
     ############################################################################
     
     
     Post_Sigma_mu = sample_omega_mu(A = A_chain_new[,,k], a_tilde, U_tilde, m_0_tilde, a_mu_tilde)
     
     Sigma_post = Post_Sigma_mu$Sigma
     mu_post    = Post_Sigma_mu$mu
     
     Sigma_chain_new[,,k]= Sigma_post 
     mu_chain_new[,,k]   = mu_post 
     
     
  }
  
  #converto da array tridimensionali a lista di matrici
  mu_chain_new_mat<-list()
  Sigma_chain_new_mat<-list()
  A_chain_new_mat<-list()
  
  for (k in 1:L) {
    mu_chain_new_mat[[k]] <- mu_chain_new[,,k]
  }
  
  for (k in 1:L) {
    Sigma_chain_new_mat[[k]] <- Sigma_chain_new[,,k]
  }
  
  for (k in 1:L) {
    A_chain_new_mat[[k]]<- A_chain_new[,,k]
  }
  
  lista_array <- list(mu_chain_new_mat, Sigma_chain_new_mat, A_chain_new_mat)
  
  return(lista_array)
  
}

pa = function (set, object){
  amat <- as(object,"matrix")
  #togliamo questa parte? 
  if (is_ugMAT(amat)) #Controlla se grafo non diretto. (GRAFO DIRETTO O NO?)
    return(NULL)
  
  pa <- names(which(amat[, set] > 0)) 
  #prendo righe i cui valori nella colonna set sono >0
  
  #Non vogliamo toglierlo perchè specifica di non avere autoanelli 
  pa <- setdiff(pa, set)
  #setdiff(x,y) prende valori che sono in x e non in y. Sto togliendo i figli dal set? (tendendo solo i genitori)
  if (length(pa)) 
    as.numeric(pa) #torna vettore con numeri 
  else NULL
}
#Sto prendendo i genitori del figlio. Set= è un figlio
#avremo un vettore che contiene le righe dove ho almeno un collegamento

fa = function(set, object){
  as.numeric(c(set, pa(set, object)))
}
n.edge = function(A){
  length(which(A[lower.tri(A)] == 1 | t(A)[lower.tri(A)] == 1))
}

names   = c("action","test","x","y")
actions = c("id","dd","rd")

library(gRbase)

# types are then indexed by (1,2,3) (respectively insert, delete and reverse a directed edge)

move = function(A, q = q){
  
  A_na = A
  diag(A_na) = NA
  
  id_set = c()
  dd_set = c()
  rd_set = c()
  
  # set of nodes for id (insert directed edge)
  
  set_id = which(A_na == 0, TRUE)
  
  if(length(set_id) != 0){
    id_set = cbind(1, rbind(set_id))
  }
  
  # set of nodes for dd (delete directed edge)
  
  set_dd = which(A_na == 1, TRUE)
  
  if(length(set_dd != 0)){
    dd_set = cbind(2, set_dd)
  }
  
  # set of nodes for rd (reverse directed edge)
  
  set_rd = which(A_na == 1, TRUE)
  
  if(length(set_rd != 0)){
    rd_set = cbind(3, set_rd)
  }
  
  O = rbind(id_set, dd_set, rd_set)
  
  repeat {
    
    i = sample(dim(O)[1],1)
    
    act_to_exe  = paste0(actions[O[i,1]],"(A=A,c(",as.vector(O[i,2]),",",as.vector(O[i,3]),"))")
    A_succ      = eval(parse(text = act_to_exe))
    act_to_eval = paste0("is.DAG(A_succ)")
    val = eval(parse(text = act_to_eval))
    
    if (val != 0){
      break
    }
  }
  
  A_new = A_succ
  
  return(list(A_new = A_new, type.operator = O[i,1], nodes = O[i,2:3]))
  
}

# 3 actions

# A     adjacency matrix of DAG D
# x,y   nodes involved in the action

id = function(A, nodes){
  x = nodes[1]
  y = nodes[2]
  A[x,y] = 1
  return(A)
}

dd = function(A, nodes){
  x = nodes[1]
  y = nodes[2]
  A[x,y] = 0
  return(A)
}

rd = function(A, nodes){ # reverse D x -> y
  x = nodes[1]
  y = nodes[2]
  A[x,y] = 0 
  A[y,x] = 1
  return(A)
}



norm_const_j = function(j, A, a, U, m, a_mu){
  
  q = ncol(A)
  
  # j:  a node (j = 1, ..., q)
  
  pa_j  = pa(j, A)
  
  
  # prior/posterior hyper-parameters of the prior induced on node j
  
  p_j  = length(pa_j)
  aD_j = a - q + p_j + 1
  
  
  if(length(pa_j) == 0){
    
    U_jj = U[j,j]
    
    const_j = - lgamma(aD_j/2) + (aD_j/2)*log(U_jj/2)
    
    
  }else{
    
    U_jj = U[j,j] - U[j,pa_j]%*%solve(U[pa_j,pa_j])%*%U[pa_j,j]
    
    const_j = - lgamma(aD_j/2) + (aD_j/2)*log(U_jj/2) + 0.5*log(det(as.matrix(U[pa_j,pa_j])))
    
  }
  
  return(const_j)
  
}

sample_omega_mu = function(A, a, U, m, a_mu){
  
  q = ncol(A) #A è il grafo
  
  # A:  a DAG, represented by its (q,q) adjacency matrix
  # S:  number of MCMC iterations
  
  sample_chol_eta_j = function(j){
    
    # j:  a node (j = 1, ..., q)
    
    pa_j  = pa(j, A)
    
    # set node hyper-parameters
    
    p_j  = length(pa_j) #numero dei genitori 
    aD_j = a - q + p_j + 1 #dato 
    
    L_j = matrix(0, 1, ncol = p_j)
    #matrice di una riga con tante colonne quanti i genitori
    #(una per ogni genitore)
    if(p_j == 0){
      
      U_jj  = U[j,j]
      D_jj  = (rgamma(1, aD_j/2, U_jj/2))^(-1)
      eta_j = rnorm(1, m[j], sqrt(D_jj/a_mu))
      
    } else{
      
      U_jj  = U[j,j] - U[j,pa_j]%*%solve(U[pa_j,pa_j])%*%U[pa_j,j]
      D_jj  = (rgamma(1, aD_j/2, U_jj/2))^(-1)
      L_j   = rmvnorm(1, -solve(U[pa_j,pa_j])%*%U[pa_j,j], D_jj*solve(U[pa_j,pa_j]))
      eta_j = rnorm(1, m[j] + L_j%*%m[pa_j], sqrt(D_jj/a_mu))
      
    }
    
    return(list(L_j = L_j, D_jj = D_jj, eta_j = eta_j))
    
  }
  
  
  chol_nodes = lapply(X = 1:q, FUN = sample_chol_eta_j)
  #resistuisce una lista. Ciclo for per tutti i nodi, restituisce una lista con
  #L_{j} ecc
  j_parents  = sapply(X = 1:q, FUN = pa, object = A)
  #restituisce se possibile un array.
  
  L   = matrix(0, q, q); diag(L) = 1 #diagonale? 
  D   = matrix(0, q, q)
  eta = c()
  
  for(j in 1:q){
    
    L[unlist(j_parents[j]),j] = chol_nodes[[j]]$L_j
    #prendo i genitori rispetto alla colonna j
    #e applico funzione che mi fa riparametrizzazione
    D[j,j] = chol_nodes[[j]]$D_jj
    eta[j] = chol_nodes[[j]]$eta_j
    
  }
  
  #riporto a mu e sigma 
  Sigma  = t(solve(L))%*%D%*%solve(L)
  mu     = t(solve(L))%*%eta
  
  
  return(list(Sigma = Sigma, mu = mu))
  
}
  

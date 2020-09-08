#Group 1
#Fredrik Gustafsson
#WheresCroc game in which the agent tries to find the target using markov model

library(WheresCroc)
#Initial vector where we use that we know that croc can't start in the positions 
#where the ranger or tourists are otherwise we just guess
initial= function(position){
  istate =matrix(0,nrow=1,ncol=40)
  istate[3:40] = 1/37
  return (istate)
}

#Copied from the provided WheresCroc.R, since I for some reason can't seem to pass it in v. 1.2.2
getOptions=function(point,edges) {
  c(edges[which(edges[,1]==point),2],edges[which(edges[,2]==point),1],point)
}

#transition matrix creates an empty matrix of 40x40 and then fill it with values
#note that column i is the probability for waterhole i
transMat = function(edges){
  
  t = matrix(0,nrow=40,ncol=40)
  
  for(i in 1:nrow(t)){
    options = getOptions(i,edges)
    transP = 1/length(options)
    
    for(j in 1:length(options)){
      t[i,options[j]] = transP
    }
  }
  return (t)
}

#check if tourist is eaten
touristEaten = function(moveInfo, positions) {
  #a tourist is eaten
  if (!is.na(positions[1]) && positions[1] < 0) {
    croc_location = abs(positions[1])
    state = matrix(0,nrow=1,ncol=40)
    state[1, croc_location] = 1
    return (state)
  }
  
  if (!is.na(positions[2]) && positions[2] < 0) {
    croc_location = abs(positions[2])
    state = matrix(0,nrow=1,ncol=40)
    state[1, croc_location] = 1
    return (state)
  }
  #no toursit eaten
  return(moveInfo$mem$state) #after HMM transition
}


#calculate next state with markov chain
nextState = function(curr_state, t, readings, probs, positions){
  probt = c(rep(0,40))
  
  for(i in 1:40){
    probsa = dnorm(readings[1],probs$salinity[i,1],probs$salinity[i,2], FALSE)
    probp = dnorm(readings[2],probs$phosphate[i,1],probs$phosphate[i,2], FALSE)
    probn = dnorm(readings[3],probs$nitrogen[i,1],probs$nitrogen[i,2], FALSE)
    probt[i] = probsa * probp * probn
  }
  
  for(i in head(positions,2)){
    if(!is.na(i) && i > 0)
      probt[i] = 0
  }
  
  trans = curr_state %*% t %*% diag(probt)
  
  #normalize
  normTran = trans/sum(trans)
  #if we divide by 0
  if(NaN %in% normTran){
    return(trans)
  }
  return (normTran)
}

#breadth first search based on code at wikipedia
bfs = function(start, goal, edges) {
  if(start == goal){
    return (c(0,0))
  }
  open = c(start)
  visit =c()
  path = list()

  while(length(open)>0){
    #retrieve node and delete from que
    curr = head(open,1)
    open = tail(open,-1)
    #frontier
    frontier = setdiff(setdiff(getOptions(curr,edges),visit),curr)
    for(i in frontier) {
      path[as.character(i)] = curr
      if(i == goal) {
        return(createPath(start, goal, path))
      }
      open = append(open,i)
      visit = append(visit, i)
    }
  }
  stop(paste("no path between ", start, "-> ", goal, "found"))
}

createPath = function(first, end, paths){
  path = c(0,end)
  curent = end
  while(curent != first){
    parent =paths[as.character(curent)]
    path = append(path,parent)
    curent = parent
  }
  return (tail(rev(path),-1)) # we create the path from end to first but want to follow from first to end, -1 allowed two actions
}

myFunction = function(moveInfo, readings, positions, edges, probs) {
  state = 1
  
  #first time
  if(length(moveInfo$moves) == 0){
    state =initial(positions)
    t = transMat(edges)
    moveInfo$mem$t = t
    #All other times
  } else {
    state  = touristEaten(moveInfo, positions)
    }
   
  moveInfo$mem$state = nextState(state, moveInfo$mem$t, readings, probs, positions)
  croc = which.max(moveInfo$mem$state)
  path = bfs(positions[3],croc,edges)
  moveInfo$moves = head(path, 2)
  return (moveInfo) 
}
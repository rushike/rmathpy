import json

def fsMemoizer(f):
    cx = {}
    def f2(*args):
        try:
            key= json.dumps(args)
        except:
            key =json.dumps(args[:-1] + (sorted(list(args[-1])),))
        if key not in cx:
            cx[key] = f(*args)
        return cx.get(key)
    return f2

def towers3(ndisks,start=1,target=3,peg_set=set([1,2,3])):
   if ndisks == 0 or start == target: 
      return [] 
   my_move = "%s --> %s"%(start,target) 
   if ndisks == 1: 
      return [my_move]
   helper_peg = peg_set.difference([start,target]).pop()
   moves_to_my_move = towers3(ndisks-1,start,helper_peg)
   moves_after_my_move = towers3(ndisks-1,helper_peg,target)
   return moves_to_my_move + [my_move] + moves_after_my_move

@fsMemoizer
def FrameStewartSolution(ndisks, start=1, end=4, pegs=set([1,2,3,4])):
    if ndisks == 0 or start == end:
        return []
    if  ndisks == 1 and len(pegs) > 1: 
        return ["%s --> %s"%(start,end)]  
    if len(pegs) == 3: 
        return towers3(ndisks,start,end,set(pegs))
    if len(pegs) >= 3 and ndisks > 0:
        best_solution = float("inf")
        best_score = float("inf")
        for kdisks in range(1,ndisks):
            helper_pegs = list(pegs.difference([start,end]))
            LHSMoves = FrameStewartSolution(kdisks,start,helper_pegs[0],pegs)
            pegs_for_my_moves = pegs.difference([helper_pegs[0]]) # cant use the peg our LHS stack is sitting on
            MyMoves = FrameStewartSolution(ndisks-kdisks,start,end,pegs_for_my_moves) #misleading variable name but meh 
            RHSMoves = FrameStewartSolution(kdisks,helper_pegs[0],end,pegs)#move the intermediat stack to 
            if any(move is None for move in [LHSMoves,MyMoves,RHSMoves]):continue #bad path :(
            move_list = LHSMoves + MyMoves + RHSMoves
            if(len(move_list) < best_score):
                best_solution = move_list
                best_score = len(move_list)
        if best_score < float("inf"):       
            return best_solution
    
    return None

N = int(input("Enter the number of disks : "))
T = int(input("Enter the number of towers : "))

res = FrameStewartSolution( N, 1, T,  set(range(1, T + 1)))
print("\n".join(res))
nn(caviar_net, [Video, P, T], SE, [active, inactive, walking, running]) :: happensAt(Video, P, T, SE).

holdsAt(Video, F, T) :- previous(T1, T), initiatedAt(F, Video, T1).
holdsAt(Video, F, T) :- previous(T1, T), previous_step(Video, F, T1), \+terminatedAt(F, Video, T1).

initiatedAt(moving(P1, P2), Video, T) :- 
    happensAt(Video, P1, T, walking),
    happensAt(Video, P2, T, walking),
    close(Video, P1, P2, T, 40).

terminatedAt(moving(P1, P2), Video, T) :- 
    happensAt(Video, p1, T, active), happensAt(Video, p1, T, active).

terminatedAt(moving(P1, P2), Video, T) :- 
    happensAt(Video, P2, T, active), happensAt(Video, P2, T, inactive).

terminatedAt(moving(P1, P2), Video, T) :- 
    happensAt(Video, P2, T, inactive), happensAt(Video, P2, T, active).


% terminatedAt(meeting(P1, P2), Video, T) :- 
%     happensAt(Video, P2, T, walking).
% 
% terminatedAt(meeting(P1, P2), Video, T) :- 
%     happensAt(Video, P2, T, walking).

previous(T1, T) :- 
    T >= 0, 
    T1 is T-1, 
    T1 >= 0.

close(Video, P1, P2, T, D) :- distance(Video, P1, P2, T, D1), D1 =< D. 

0.0::previous_step(tensor(train(0)),moving(p1,p2),0).
distance(tensor(train(0)),p1,p2,0,0).

:-consult('caviar_deepproblog/data/vision_data/spatial_info.pl').

nn(caviar_cnn, [Video, P, T], SE, [active, inactive, walking, running]) :: happensAt(Video, P, T, SE).

holdsAt(Video, F, T) :- previous(T1, T), initiatedAt(F, Video, T1).
holdsAt(Video, F, T) :- previous(T1, T), previous_step(Video, F, T1), \+terminatedAt(F, Video, T1).

initiatedAt(meeting(P1, P2), Video, T) :- 
    happensAt(Video, P1, T, active),
    happensAt(Video, P2, T, active),
    is_close(Video, P1, P2, T, 25).

initiatedAt(meeting(P1, P2), Video, T) :- 
    happensAt(Video, P1, T, active),
    happensAt(Video, P2, T, inactive),
    is_close(Video, P1, P2, T, 25).

initiatedAt(meeting(P1, P2), Video, T) :- 
    happensAt(Video, P1, T, inactive),
    happensAt(Video, P2, T, active),
    is_close(Video, P1, P2, T, 25).

initiatedAt(meeting(P1, P2), Video, T) :- 
    happensAt(Video, P1, T, inactive),
    happensAt(Video, P2, T, inactive),
    is_close(Video, P1, P2, T, 25).

terminatedAt(meeting(P1, P2), Video, T) :- 
    happensAt(Video, P1, T, running).

terminatedAt(meeting(P1, P2), Video, T) :- 
    happensAt(Video, P2, T, running).

terminatedAt(meeting(P1, P2), Video, T) :- 
    happensAt(Video, P1, T, walking), far(Video, P1, P2, T, 25).

terminatedAt(meeting(P1, P2), Video, T) :- 
    happensAt(Video, P2, T, walking), far(Video, P1, P2, T, 25).


previous(T1, T) :- 
    T >= 0, 
    T1 is T-1, 
    T1 >= 0.

is_close(Video, P1, P2, T, D) :- distance(Video, P1, P2, T, D1), D1 =< D. 
far(Video, P1, P2, T, D) :- distance(Video, P1, P2, T, D1), D1 > D. 

0.0::previous_step(tensor(train(0)),meeting(p1,p2),0).
distance(tensor(train(0)),p1,p2,0,0).

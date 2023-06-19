:-consult('cached_predicates.pl').

nn(lstm_net, [Video, P, T], SE, [active, inactive, walking, running]) :: happensAt(Video, P, T, SE).

holdsAt(Video, T, F) :- previous(T1, T), initiatedAt(F, Video, T1).
holdsAt(Video, T, F) :- previous(T1, T), cached(Video, F, T1), \+terminatedAt(F, Video, T1).

%holdsAt(Video, P1, P2, T, moving) :- holdsAt(Video, moving(P1,P2), T).
%holdsAt(Video, P1, P2, T, meeting) :- holdsAt(Video, meeting(P1,P2), T).
%holdsAt(Video, P1, P2, T, nointeraction) :- holdsAt(Video, nointeraction(P1,P2), T).


%moving complex event
initiatedAt(moving(P1, P2), Video, T):- 
    happensAt(Video, P1, T, walking),
    happensAt(Video, P2, T, walking),
    orientationMove(P1, P2, T),
    close(P1, P2, 34, T).
    
%terminating conditions
terminatedAt(moving(P1,P2), Video, T) :- happensAt(Video, P1, T, walking), far(P1,P2,34,T).
terminatedAt(moving(P1,P2), Video, T) :- happensAt(Video, P2, T, walking), far(P1,P2,34,T).
terminatedAt(moving(P1,P2),T) :- happensAt(Video, P1, T, active), happensAt(Video, P2, T, active).
terminatedAt(moving(P1,P2),T) :- happensAt(Video, P1, T, active), happensAt(Video, P2, T, inactive).
terminatedAt(moving(P1,P2),T) :- happensAt(Video, P1, T, inactive), happensAt(Video, P2, T, active).
terminatedAt(moving(P1,P2),T) :- happensAt(Video, P1, T, running), happensAt(Video, P2, T, _).
terminatedAt(moving(P1,P2),T) :- happensAt(Video, P1, T, _), happensAt(Video, P2, T, running).

%meeting complex event
initiatedAt(meeting(P1,P2), Video, T) :- 
    happensAt(Video, P1, T, active), 
    happensAt(Video, P2, T, active),
    close(P1,P2,25,T).

initiatedAt(meeting(P1,P2), Video, T) :- 
    happensAt(Video, P1, T, active), 
    happensAt(Video, P2, T, inactive),
    close(P1,P2,25,T).
    
initiatedAt(meeting(P1,P2), Video, T) :- 
    happensAt(Video, P1, T, inactive), 
    happensAt(Video, P2, T, active),
    close(P1,P2,25,T).

initiatedAt(meeting(P1,P2), Video, T) :- 
    happensAt(Video, P1, T, inactive), 
    happensAt(Video, P2, T, inactive),
    close(P1,P2,25,T).

%terminating conditions
terminatedAt(meeting(P1,P2), Video, T) :- happensAt(Video, P1, T, running), happensAt(Video, P2, T, _).
terminatedAt(meeting(P1,P2), Video, T) :- happensAt(Video, P1, T, _), happensAt(Video, P2, T, running).
terminatedAt(meeting(P1,P2), Video, T) :- happensAt(Video, P1, T, walking), far(P1,P2,25,T).
terminatedAt(meeting(P1,P2), Video, T) :- happensAt(Video, P2, T, walking), far(P1,P2,25,T).

%nointeraction complex event
initiatedAt(nointeraction(P1,P2), Video, T) :- \+holdsAt(Video, meeting(P1,P2), T), \+holdsAt(Video, moving(P1,P2),T).
terminatedAt(nointeraction(P1,P2), Video, T) :- holdsAt(Video, meeting(P1,P2), T).
terminatedAt(nointeraction(P1,P2), Video, T) :- holdsAt(Video, moving(P1,P2), T).

previous(T1, T) :- 
    T >= 0, 
    T1 is T-1, 
    T1 >= 0.


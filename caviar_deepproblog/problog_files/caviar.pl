nn(caviar_net, [Video, P, T], SE, [active, inactive, walking, running]) :: happensAt(Video, P, T, SE).

holdsAt(Video, F, T) :- previous(T1, T), initiatedAt(F, Video, T1).
holdsAt(Video, F, T) :- previous(T1, T), previous_step(Video, F, T1), \+terminatedAt(F, Video, T1).

%moving complex event

initiatedAt(moving(P1, P2), Video, T):-
    happensAt(Video, P1, T, walking),
    happensAt(Video, P2, T, walking),
    orientationMove(Video, P1, P2, T),
    is_close(Video, P1, P2, T, 34).

terminatedAt(moving(P1,P2), Video, T) :-
    happensAt(Video, P1, T, walking), is_far(Video, P1, P2, T, 34).

terminatedAt(moving(P1,P2), Video, T) :-
    happensAt(Video, P2, T, walking), is_far(Video, P1, P2, T, 34).

terminatedAt(moving(P1,P2),T) :-
    happensAt(Video, P1, T, active), happensAt(Video, P2, T, active).

terminatedAt(moving(P1,P2),T) :-
    happensAt(Video, P1, T, active), happensAt(Video, P2, T, inactive).

terminatedAt(moving(P1,P2),T) :-
    happensAt(Video, P1, T, inactive), happensAt(Video, P2, T, active).

terminatedAt(moving(P1,P2),T) :-
    happensAt(Video, P1, T, running).

terminatedAt(moving(P1,P2),T) :-
    happensAt(Video, P2, T, running).


%meeting complex event

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
    happensAt(Video, P1, T, walking), is_far(Video, P1, P2, T, 25).

terminatedAt(meeting(P1, P2), Video, T) :-
    happensAt(Video, P2, T, walking), is_far(Video, P1, P2, T, 25).


%nointeraction complex event
% initiatedAt(nointeraction(P1,P2), Video, T) :-
%    \+holdsAt(Video, meeting(P1,P2), T), \+holdsAt(Video, moving(P1,P2),T).

% terminatedAt(nointeraction(P1,P2), Video, T) :-
%    holdsAt(Video, meeting(P1,P2), T).

% terminatedAt(nointeraction(P1,P2), Video, T) :-
%    holdsAt(Video, moving(P1,P2), T).

previous(T1, T) :-
    T >= 0,
    T1 is T-1,
    T1 >= 0.

is_close(Video, P1, P2, T, D) :- distance(Video, P1, P2, T, D1), D1 =< D.
is_far(Video, P1, P2, T, D) :- distance(Video, P1, P2, T, D1), D1 > D.
orientationMove(Video, P1, P2, T) :- orientation(Video, P1, P2, T, D), D =< 45.

0.0::previous_step(tensor(train(0)),meeting(p1,p2),0).
distance(tensor(train(0)),p1,p2,0,0).
orientation(tensor(train(0)),p1,p2,0,0).
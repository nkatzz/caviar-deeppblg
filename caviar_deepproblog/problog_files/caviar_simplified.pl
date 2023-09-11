:- consult('/home/yuzer/ncsr/dpl-benchmark/caviar-deepproblog/caviar_deepproblog/data/vision_data/spatial_info.pl').

nn(caviar_cnn, [Video, P, T], SE, [active, inactive, walking, running]) :: happensAt(Video, P, T, SE).

holdsAt(Video, meeting(P1, P2), T) :-
    happensAt(Video, P1, T, active),
    happensAt(Video, P2, T, active),
    is_close(Video, P1, P2, T, 25).

holdsAt(Video, meeting(P1, P2), T) :-
    happensAt(Video, P1, T, inactive),
    happensAt(Video, P2, T, inactive),
    is_close(Video, P1, P2, T, 25).


is_close(Video, P1, P2, T, D) :- distance(Video, P1, P2, T, D1), D1 =< D.
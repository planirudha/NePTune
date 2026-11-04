% Scene: obj/7 facts -> object/1, position/3, has_*/2; initial state/2.
% Rules are verbatim from kdasp/perfect_theory.lp.
scene(X) :- scene(X,_).
object(ID) :- obj(ID,_,_,_,_,_,_).
position(ID,X,Y) :- obj(ID,X,Y,_,_,_,_).
has_size(ID,SIZE) :- obj(ID,_,_,_,_,_,SIZE).
has_color(ID,COLOR) :- obj(ID,_,_,_,COLOR,_,_).
has_material(ID,MATERIAL):- obj(ID,_,_,MATERIAL,_,_,_).
has_shape(ID,SHAPE) :- obj(ID,_,_,_,_,SHAPE,_).
state(0,ID) :- object(ID).
state(T+1,ID) :- scene(T), object(ID).

% Attributes: query_*, same_* and equal_{size,color,material,shape}.
% Rules are verbatim from kdasp/perfect_theory.lp.
size(T+1,SIZE) :- query_size(T), state(T,ID), has_size(ID,SIZE).
color(T+1,COLOR) :- query_color(T), state(T,ID), has_color(ID,COLOR).
material(T+1,MATERIAL) :- query_material(T), state(T,ID), has_material(ID,MATERIAL).
shape(T+1,SHAPE) :- query_shape(T), state(T,ID), has_shape(ID,SHAPE).
state(T+1,ID') :- same_size(T), state(T,ID), has_size(ID,SIZE), has_size(ID',SIZE), ID!=ID'.
state(T+1,ID') :- same_color(T), state(T,ID), has_color(ID,COLOR), has_color(ID',COLOR), ID!=ID'.
state(T+1,ID') :- same_material(T), state(T,ID), has_material(ID,MATERIAL), has_material(ID',MATERIAL), ID!=ID'.
state(T+1,ID') :- same_shape(T), state(T,ID), has_shape(ID,SHAPE), has_shape(ID',SHAPE), ID!=ID'.
bool(T+1,yes) :- equal_size(T,T'), size(T,V), size(T',V'), V=V'.
bool(T+1,no) :- equal_size(T,T'), not bool(T+1,yes).
bool(T+1,yes) :- equal_color(T,T'), color(T,V), color(T',V'), V=V'.
bool(T+1,no) :- equal_color(T,T'), not bool(T+1,yes).
bool(T+1,yes) :- equal_material(T,T'), material(T,V), material(T',V'), V=V'.
bool(T+1,no) :- equal_material(T,T'), not bool(T+1,yes).
bool(T+1,yes) :- equal_shape(T,T'), shape(T,V), shape(T',V'), V=V'.
bool(T+1,no) :- equal_shape(T,T'), not bool(T+1,yes).

rule recommend
match s : UsersMovies!User
with  t : UsersMovies!Movie
{
  compare
  {
    return selectOne(s).rate->collect(e | selectOne(e).ratings)->flatten()->collect(e | selectOne(e).ratings)->flatten()->size();
  }
}
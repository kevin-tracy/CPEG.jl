function density(r::T) where T
      R_MARS = 3386200.0
      h = norm(r) - R_MARS
      print(h)
end

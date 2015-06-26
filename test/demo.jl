
using Qwt
using QuickStructs

function cpos(t)
  hyp = t
  θ = t * π / 20.
  x = cos(θ) * hyp
  y = sin(θ) * hyp
  P2(x,y)
end

function demo_anim()
  
  scene = currentScene()
  empty!(scene)
  background!(:gray)

  c = circle!(10)
  brush!(c, :black)

  T = 0.0 : 1.0 : 400.0
  for t in T
    position!(c, cpos(t))
    sleep(0.002)
  end

  scene
end


function demo_anim_lsm()
  
  scene = currentScene()
  empty!(scene)
  background!(:white)

  # create circles
  n = 100
  # colors = [makecolor(0,0,0,0) for i in 1:n]
  circles = SceneItem[pen!(brush!(circle!(10), :white), 0) for i in 1:n]

  T = 0.0 : 1.0 : 400.0
  for (i,t) in enumerate(T)

    # adjust position/alpha for circles
    for (j,c) in enumerate(circles)
      j > i && break 
      tj = T[i-j+1]
      alpha = max(0., min(exp((tj-t) / 10), 1.0))
      # brush!(c, 0, 0, 0, alpha)
      # alpha!(c, alpha)
      # colors[j][:setAlphaF](alpha)
      # brush!(c, colors[j])
      brush!(c, 1-alpha, 1-alpha, 1-alpha)
      position!(c, cpos(tj))
    end

    sleep(0.000001)
  end
  
  scene
end



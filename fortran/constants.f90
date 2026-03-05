module constants
 implicit none
 real(8),parameter:: pi = acos(-1d0), clight = 2.99792458d10
 real(8),parameter:: day = 24d0*3600d0, year=day*365.25d0
 real(8),parameter:: msun = 1.989d33, lsun=3.839d33, sigma=5.67d-5

contains

 elemental function temperature(l,r) result(T)
  real(8),intent(in)::l,r
  real(8)::T
  T = (max(l,1d-10)/(4d0*pi*sigma*r**2))**0.25d0
 end function temperature

 function intpol(xp,x,y) result(yp)
  real(8),intent(in):: xp,x(1:2),y(1:2)
  real(8):: yp
  yp = (y(1)*(x(2)-xp)+y(2)*(xp-x(1)))/(x(2)-x(1))
 end function intpol

end module constants

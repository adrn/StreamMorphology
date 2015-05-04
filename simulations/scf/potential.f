C***********************************************************************
C
C
                          SUBROUTINE potparams
C
C
C***********************************************************************
C
C
C     Subroutine to read in potential parameters.
C
C     Input parameters:
C
C
C
C=======================================================================

      INCLUDE 'scf.h'

      CHARACTER *1 pcomment
      CHARACTER*8 filepar,filename
C=======================================================================


      filepar = potparsfile
      filename = filepar(1:6)

      OPEN(UNIT=upotpars,FILE=filename,STATUS='OLD')

C     Read parameters, close the file.
C     --------------------------------

      READ(upotpars,'(a)') pcomment
      READ(upotpars,*) myn_a
      READ(upotpars,*) myn_b
      READ(upotpars,*) myn_M
      READ(upotpars,'(a)') pcomment
      READ(upotpars,*) hqs_c
      READ(upotpars,*) hqs_M
      READ(upotpars,'(a)') pcomment
      READ(upotpars,*) nfw_rs
      READ(upotpars,*) nfw_vh
      READ(upotpars,*) nfw_a
      READ(upotpars,*) nfw_b
      READ(upotpars,*) nfw_c
      READ(upotpars,*) nfw_phi
      READ(upotpars,*) nfw_theta
      READ(upotpars,*) nfw_psi
      CLOSE(UNIT=upotpars)

      RETURN
      END

C***********************************************************************
C
C
                          SUBROUTINE accp_external
C
C
C***********************************************************************
C Triaxial NFW halo (using the Lee & Suto (2003) approximation)
C
      INCLUDE 'scf.h'

C   Declaration of local variables.
C   -------------------------------
      INTEGER i
      REAL*8 r2,z2,rad,tsrad,sqz2b2,tdr,
     &         tdz,phim,phis,phih,
     &         tu,gee,msun,cmpkpc,secperyr,
     &         a,b,hs,vcirc2,GMs,GM,
     &         mvir,rvir,cee,rs,mhalo,phi0,mr,p,
     &         yor2,zor2,axh,ayh,azh,
     &         xx,yy,zz

      PARAMETER(gee=6.673840e-8,
     &          msun=1.9890999999999997e33,
     &          cmpkpc=3.085677581467192e21,
     &          secperyr=3.15576d7)

      LOGICAL firstc
      DATA firstc/.TRUE./
      SAVE firstc,vcirc2,a,b,GM,GMs,hs,tu,rs,mhalo,phi0

C -----------------------------------------------------------------------
C       Three Component Galaxy model:
C               Miyamoto-Nagai Disk + Hernquist Spheroid + NFW Halo
C --------------------------------------------------------------

C Convert parameters to appropriate units....
      IF(firstc)THEN
            firstc = .FALSE.

C           Simulation units
            tu = SQRT((cmpkpc*ru)**3/(msun*mu*gee))
            vu = (ru*cmpkpc*1.e-5)/tu
            WRITE(6,*)ru,tu/secperyr,vu

            strength = 1.
            IF(ntide.GT.0) THEN
                  strength=0.
C
                  xframe = xframe/ru
                  yframe = yframe/ru
                  zframe = zframe/ru
                  vxframe = vxframe/vu
                  vyframe = vyframe/vu
                  vzframe = vzframe/vu
            END IF
C
            DO 5 i=1,nbodies
              vx(i)=vx(i)+vxframe
              vy(i)=vy(i)+vyframe
              vz(i)=vz(i)+vzframe
 5          CONTINUE

C           Miyamoto-Nagai parameters
            a = myn_a/ru
            b = myn_b/ru
            GM = myn_M/mu

C           Hernquist parameters
            GMs = hqs_M/mu
            hs = hqs_c/ru

C           NFW halo parameters
            nfw_rs = nfw_rs/ru
            nfw_vh = nfw_vh/vu
            mhalo = rs*phi0/G

      END IF

      DO 10 i=1,nbodies

      xx = (x(i)+xframe)
      yy = (y(i)+yframe)
      zz = (z(i)+zframe)

C     Compute potential, acceleration due to spheroid
      r2 = xx*xx + yy*yy
      z2 = zz*zz
      rad = sqrt(r2+z2)
      tsrad = GMs/(rad+hs)**2/rad
      phis = -GMs/(rad+hs)

      ax(i) = ax(i) - strength*tsrad*xx
      ay(i) = ay(i) - strength*tsrad*yy
      az(i) = az(i) - strength*tsrad*zz

      potext(i) = strength*phis

C     Compute potential, acceleration due to disk
      sqz2b2 = sqrt(z2 + b*b)
      tdr = GM/(r2 + (a + sqz2b2)**2)**1.5
      tdz = tdr*(a/sqz2b2 + 1.)
      phim = -GM/sqrt(r2+(a+sqz2b2)**2)

      ax(i) = ax(i) - strength*tdr*xx
      ay(i) = ay(i) - strength*tdr*yy
      az(i) = az(i) - strength*tdz*zz
      potext(i) = potext(i) + strength*phim

C     Compute potential, acceleration due to halo
      CALL accp_trix(i,rad,xx,yy,zz,
     &               phih,axh,ayh,azh)

      ax(i) = ax(i) + strength*axh
      ay(i) = ay(i) + strength*ayh
      az(i) = az(i) + strength*azh

      potext(i) = potext(i) + strength*phih

 10     CONTINUE

      RETURN
      END

C***********************************************************************
C
C
      SUBROUTINE accp_trix(k,r,xf,yf,zf,
     &                     phih,axh,ayh,azh)
C
C
C***********************************************************************
C A subroutine to calculate acceleration due to triaxial NFW halo
C Following Jing & Suto 2002 and Lee & Suto 2003 formalism

C   Declaration of local variables.
C   -------------------------------
      INCLUDE 'scf.h'
      INTEGER k
      REAL*8 xx,yy,zz,xf,yf,zf
      REAL*8 phi,theta,psi
      REAL*8 eb2,ec2,phi0
      REAL*8 phih,p,r,yor2,zor2,axh,ayh,azh,ax_h,ay_h,az_h
      REAL*8 coa,boa,cob
      REAL*8 R11,R12,R13,R21,R22,R23,R31,R32,R33
      REAL*8 x0,x1,x2,x4,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,
     &       x16,x17,x18,x19,x20,x21,x22

      SAVE boa,coa,eb2,ec2,c2,phi0,
     &     R11,R12,R13,R21,R22,R23,R31,R32,R33

C -----------------------------------------------------------------------
      IF(k.EQ.1)THEN
            phi = nfw_phi
            theta = nfw_theta
            psi = nfw_psi

            boa = nfw_b/nfw_a
            coa = nfw_c/nfw_a

            eb2 = 1.0d0 - boa*boa
            ec2 = 1.0d0 - coa*coa

            R11 = -sin(phi)*sin(psi)*cos(theta) + cos(phi)*cos(psi)
            R12 = sin(phi)*cos(psi) + sin(psi)*cos(phi)*cos(theta)
            R13 = sin(psi)*sin(theta)
            R21 = -sin(phi)*cos(psi)*cos(theta) - sin(psi)*cos(phi)
            R22 = -sin(phi)*sin(psi) + cos(phi)*cos(psi)*cos(theta)
            R23 = sin(theta)*cos(psi)
            R31 = sin(phi)*sin(theta)
            R32 = -sin(theta)*cos(phi)
            R33 = cos(theta)

            phi0 = nfw_vh*nfw_vh
      ENDIF

      p = r/nfw_rs
      yor2 = yf*yf/rad/rad
      zor2 = zf*zf/rad/rad

      xx = R11*xf + R12*yf + R13*zf
      yy = R21*xf + R22*yf + R23*zf
      zz = R31*xf + R32*yf + R33*zf

      x0 = r + nfw_rs
      x1 = x0**2
      x2 = phi0/(12*r**7*x1)
      x4 = yy*yy
      x6 = zz*zz
      x7 = eb2*x4 + ec2*x6
      x8 = nfw_rs**2
      x9 = 6*x8
      x10 = DLOG(x0/nfw_rs)
      x11 = x0*x10
      x12 = 3*nfw_rs
      x13 = r*x12
      x14 = xx*xx
      x15 = x13 - x14 - x4 - x6
      x16 = x15 + x9
      x17 = 6*nfw_rs*x0*(r*x16 - x11*x9)
      x18 = x1*x10
      x19 = r**2
      x20 = x0*x19
      x21 = 2*r*x0
      x22 = -12*r**5*nfw_rs*x0 + 12*x19*x19*nfw_rs*x18 +
     & x12*x7*(x16*x19 - 18*x18*x8 + x20*(2*r - x12) + x21*
     & (x15 + 9*x8)) - x20*(eb2 + ec2)*(-6*r*nfw_rs*(x19 - x8) +
     & 6*nfw_rs*x11*(x19 - 3*x8) + x20*(-4*r + x12) + x21*
     & (-x13 + 2*x14 + 2*x4 + 2*x6 + x9))

      phih = phi0*((eb2/2 + ec2/2)*((1/p - 1/p**3)*x10 - 1 + (2*x19/x8
     &      - 3*p + 6)/(6*x19/x8)) + (eb2*x4/(2*x19) + ec2*x6/(2*x19))
     &      *((x19/x8 - 3*p - 6)/(2*x19/x8*(p + 1)) + 3*x10/p**3) -
     &      x10/p)

      ax_h = -x2*xx*(x17*x7 + x22)
      ay_h = -x2*yy*(x17*(-x19*eb2 + x7) + x22)
      az_h = -x2*zz*(x17*(-x19*ec2 + x7) + x22)

      axh = R11*ax_h + R21*ay_h + R31*az_h
      ayh = R12*ax_h + R22*ay_h + R32*az_h
      azh = R13*ax_h + R23*ay_h + R33*az_h

      RETURN
      END

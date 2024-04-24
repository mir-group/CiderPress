module fast_sph_harm

implicit none

real(8), allocatable :: coefs(:)
integer, allocatable :: exps(:,:)
integer, allocatable :: starts(:)
integer, allocatable :: lengths(:)
integer, allocatable :: ls(:)
integer, allocatable :: xi(:,:)

integer :: lm_max_buf = -1
integer :: nbuf = -1

complex(8), allocatable :: ylm(:,:)
real(8), allocatable :: coef0(:,:)
real(8), allocatable :: coef1(:,:)
real(8), allocatable :: c0(:)
real(8), allocatable :: c1(:)

contains

    function dinf()
        double precision dinf
        dinf = 1.0d300
        dinf = dinf*dinf
    end

    subroutine lpmn(mm,m,n,x,pm,pd)
        integer, intent(in)  :: mm ! physical dimension of PM and PD
        integer, intent(in)  :: m  ! Order of Pmn(x)
        integer, intent(in)  :: n  ! Degree of Pmn(x)
        real(8), intent(in)  :: x  ! Argument of Pmn(x)
        real(8), intent(out) :: pm(0:mm,0:n) ! Pmn(x)
        real(8), intent(out) :: pd(0:mm,0:n) ! Pmn'(x)
        integer i,j,ls
        real(8) xq,xs
        intrinsic min

        do i=0,n
            do j=0,m
                pm(j,i)=0.0d0
                pd(j,i)=0.0d0
            enddo
        enddo
        pm(0,0)=1.0d0
        if (n.eq.0) return
        if (dabs(x).eq.1.0d0) then
            do i=1,n
                pm(0,i)=x**i
                pd(0,i)=0.5d0*i*(i+1.0d0)*x**(i+1)
            enddo
            do j=1,n
                do i=1,m
                    if (i.eq.1) then
                        pd(i,j)=dinf()
                    elseif (i.eq.2) then
                        pd(i,j)=-0.25d0*(j+2)*(j+1)*j*(j-1)*x**(j+1)
                    endif
                enddo
            enddo
            return
        endif
        ls=1
        if (dabs(x).gt.1.0d0) ls=-1
        xq=dsqrt(ls*(1.0d0-x*x))
        if (x.lt.-1d0) xq=-xq
        xs=ls*(1.0d0-x*x)
        do i=1,m
            pm(i,i)=-ls*(2.0d0*i-1.0d0)*xq*pm(i-1,i-1)
        enddo
        do i=0,min(m,n-1)
            pm(i,i+1)=(2.0d0*i+1.0d0)*x*pm(i,i)
        enddo
        do i=0,m
            do j=i+2,n
                pm(i,j)=((2.0d0*j-1.0d0)*x*pm(i,j-1)-  &
                        (i+j-1.0d0)*pm(i,j-2))/(j-i)
            enddo
        enddo
        pd(0,0)=0.0d0
        do j=1,n
            pd(0,j)=ls*j*(pm(0,j-1)-x*pm(0,j))/xs
        enddo
        do i=1,m
            do j=i,n
                pd(i,j)=ls*i*x*pm(i,j)/xs+(j+i)    &
                        *(j-i+1.0d0)/xq*pm(i-1,j)
            enddo
        enddo
        return
    end subroutine lpmn

    subroutine lpmn_mod(mm,m,n,x,pm,pd)
        integer, intent(in)  :: mm ! physical dimension of PM and PD
        integer, intent(in)  :: m  ! Order of Pmn(x)
        integer, intent(in)  :: n  ! Degree of Pmn(x)
        real(8), intent(in)  :: x  ! Argument of Pmn(x)
        real(8), intent(out) :: pm(0:mm,0:n) ! Pmn(x)
        real(8), intent(out) :: pd(0:mm,0:n) ! Pmn'(x)
        integer i,j,ls
        real(8) xq,xs
        intrinsic min

        do i=0,n
            do j=0,m
                pm(j,i)=0.0d0
                pd(j,i)=0.0d0
            enddo
        enddo
        pm(0,0)=1.0d0
        if (n.eq.0) return
        if (dabs(x).eq.1.0d0) then
            do i=1,n
                pm(0,i)=x**i
                pd(0,i)=0.5d0*i*(i+1.0d0)*x**(i+1)
            enddo
            do j=1,n
                do i=1,m
                    if (i.eq.1) then
                        pd(i,j)=dinf()
                    elseif (i.eq.2) then
                        pd(i,j)=-0.25d0*(j+2)*(j+1)*j*(j-1)*x**(j+1)
                    endif
                enddo
            enddo
            return
        endif
        ls=1
        if (dabs(x).gt.1.0d0) ls=-1
        xq=dsqrt(ls*(1.0d0-x*x))
        if (x.lt.-1d0) xq=-xq
        xs=ls*(1.0d0-x*x)
        do i=1,m
            pm(i,i)=-ls*(2.0d0*i-1.0d0)*pm(i-1,i-1)
        enddo
        do i=0,min(m,n-1)
            pm(i,i+1)=(2.0d0*i+1.0d0)*x*pm(i,i)
        enddo
        do i=0,m
            do j=i+2,n
                pm(i,j)=((2.0d0*j-1.0d0)*x*pm(i,j-1)-  &
                        (i+j-1.0d0)*pm(i,j-2))/(j-i)
            enddo
        enddo
        pd(0,0)=0.0d0
        do j=1,n
            pd(0,j)=ls*j*(pm(0,j-1)-x*pm(0,j))/xs
        enddo
        do i=1,m
            do j=i,n
                pd(i,j)=ls*i*x*pm(i,j)/xs+(j+i)    &
                        *(j-i+1.0d0)/xq*pm(i-1,j)
            enddo
        enddo
        return
    end subroutine lpmn_mod

    subroutine sph_harm_and_deriv( lm_max, r, res, dres, n )
        integer, intent(in)  :: lm_max
        real(8), intent(in)  :: r(3,n)
        real(8), intent(out) :: res(lm_max,n)
        real(8), intent(out) :: dres(3,lm_max,n)
        integer, intent(in)  :: n

        integer                 :: lmax, i, lm, lm0, l, m
        real(8), allocatable    :: pm(:,:), pd(:,:), fac_ml(:,:)
        real(8)                 :: f0, fpi, rt2
        complex(8), allocatable :: xyp(:)
        complex(8)              :: v, xy, dx, dy, dz

        lmax = int(sqrt(dble(lm_max)-1) + 1e-7)
        allocate( pm(0:lmax,0:lmax) )
        allocate( pd(0:lmax,0:lmax) )
        allocate( fac_ml(0:lmax,0:lmax) )
        allocate( xyp(0:lmax) )

        fpi = 16.0 * atan(1.0d0)
        f0 = 1.0d0 / dsqrt(fpi)
        rt2 = dsqrt(2.0d0)

        do l=0,lmax
            fac_ml(0,l) = dble(2*l+1) / fpi
            if (l.gt.0) then
                do m=1,l
                    fac_ml(m,l) = fac_ml(m-1,l) / dble((l+m) * (l-m+1))
                    !fac_ml(m,l) = fac_ml(m-1,l) / dble((l-m+1))
                enddo
            endif
            fac_ml(0,l) = dsqrt( fac_ml(0,l) )
            do m=1,l
                fac_ml(m,l) = rt2 * (-1.0d0)**m * dsqrt( fac_ml(m,l) )
            enddo
        enddo

        xyp(0) = cmplx(1.0d0, 0.0d0, kind=8)

        do i=1,n
            xy = cmplx(r(1,i), r(2,i), kind=8) !/ dsqrt(r(1,i)*r(1,i)+r(2,i)*r(2,i)+1e-200)
            call lpmn_mod(lmax,lmax,lmax,r(3,i),pm,pd)
            res(1,i) = f0
            dres(:,1,i) = 0.0d0
            if (lmax.eq.0) continue
            do l=1,lmax
                lm0 = l * (l+1) + 1
                xyp(l) = xyp(l-1) * xy
                res(lm0,i) = fac_ml(0,l) * pm(0,l)
                dres(1,lm0,i) = 0.0d0
                dres(2,lm0,i) = 0.0d0
                dres(3,lm0,i) = fac_ml(0,l) * pd(0,l)
                do m=1,l
                    v = fac_ml(m,l) * pm(m,l) * xyp(m)
                    !v = fac_ml(m,l) * xyp(m)
                    dz = fac_ml(m,l) * pd(m,l) * xyp(m)
                    dx = fac_ml(m,l) * m * pm(m,l) * xyp(m-1)
                    dy = fac_ml(m,l) * m * pm(m,l) * xyp(m-1)
                    lm = lm0 - m
                    res(lm,i) = aimag( v )
                    dres(1,lm,i) = aimag( dx )
                    dres(2,lm,i) = real ( dy )
                    dres(3,lm,i) = aimag( dz )
                    lm = lm0 + m
                    res(lm,i) = real ( v )
                    dres(1,lm,i) =  real ( dx )
                    dres(2,lm,i) = -aimag( dy )
                    dres(3,lm,i) =  real ( dz )
                    !if (i.eq.1) then
                    !    print *, l, m, pm(m,l), real(xyp(m)), aimag(xyp(m))
                    !    print *, r(1,i), r(2,i), r(3,i), real( v ), aimag( v ), res(lm,i)
                    !endif
                enddo
            enddo
        enddo
    end subroutine sph_harm_and_deriv

    subroutine lpmn_vec(m,n,x,nn,pm,pd)
        integer, intent(in)  :: m  ! Order of Pmn(x)
        integer, intent(in)  :: n  ! Degree of Pmn(x)
        real(8), intent(in)  :: x(nn)  ! Argument of Pmn(x)
        integer, intent(in)  :: nn     ! Dimension of x
        real(8), intent(out) :: pm(0:m,0:n,nn) ! Pmn(x)
        real(8), intent(out) :: pd(0:m,0:n,nn) ! Pmn'(x)
        integer i
        do i=1,nn
            call lpmn(m,m,n,x(i),pm(:,:,i),pd(:,:,i))
        enddo
        return
    end subroutine

    subroutine initialize_sph_harm( coefs_, exps_, starts_, lengths_, ls_, nl, nc )

        real(8), intent(in) :: coefs_(nc)
        integer, intent(in) :: exps_(3,nc)
        integer, intent(in) :: starts_(nl)
        integer, intent(in) :: lengths_(nl)
        integer, intent(in) :: ls_(nl)
        integer, intent(in) :: nl
        integer, intent(in) :: nc

        integer i,j,k,q,p,lmax

        allocate( coefs(nc) )
        allocate( exps(3,nc) )
        allocate( starts(nl) )
        allocate( lengths(nl) )
        allocate( ls(nl) )
        lmax = maxval(ls_)
        allocate( xi(lmax,nc) )

        xi(:,:) = 0

        j=1
        do i=1,nl
            starts(i) = starts_(i)
            lengths(i) = lengths_(i)
            ls(i) = ls_(i)
            ! print *, starts(i), lengths(i), ls(i)
            do k=starts(i),starts(i)+lengths(i)-1
                p=0
                if (exps_(1,k).gt.0) then
                    do q=1,exps_(1,k)
                        p = p + 1
                        xi(p,k) = 1
                    enddo
                endif
                if (exps_(2,k).gt.0) then
                    do q=1,exps_(2,k)
                        p = p + 1
                        ! print *, "py", p, q, k, exps_(2,k)
                        xi(p,k) = 2
                    enddo
                endif
                if (exps_(3,k).gt.0) then
                    do q=1,exps_(3,k)
                        p = p + 1
                        ! print *, "pz", p, q, k, exps_(3,k)
                        xi(p,k) = 3
                    enddo
                endif
            enddo
        enddo
        do i=1,nc
            coefs(i) = coefs_(i)
            exps(:,i) = exps_(:,i)
            ! print *, xi(1,i), xi(2,i), xi(3,i), xi(4,i)
        enddo

    end subroutine initialize_sph_harm

    subroutine eval_sph_harm( L, r, res, n )
        integer, intent(in) :: L
        real(8), intent(in) :: r(3,n)
        real(8), intent(out) :: res(n)
        integer, intent(in) :: n

        integer i

        res(:) = 0.0_8
        do i=starts(L+1),starts(L+1)+lengths(L+1)-1
            res(:) = res(:) + coefs(i) * r(1,:)**exps(1,i) * r(2,:)**exps(2,i) * r(3,:)**exps(3,i)
        enddo
    end subroutine eval_sph_harm

    subroutine eval_sph_harm_fast( L, r, res, n )
        integer, intent(in) :: L
        real(8), intent(in) :: r(3,n)
        real(8), intent(out) :: res(n)
        integer, intent(in) :: n

        integer i,ll

        res(:) = 0.0_8
        ll = ls(L+1)
        if (ll.eq.0) then
            res(:) = coefs(starts(L+1))
        elseif (ll.eq.1) then
            i = starts(L+1)
            res(:) = coefs(i) * r(xi(1,i),:)
        elseif (ll.eq.2) then
            do i=starts(L+1),starts(L+1)+lengths(L+1)-1
                res(:) = res(:) + coefs(i) * r(xi(1,i),:) * r(xi(2,i),:)
            enddo
        elseif (ll.eq.3) then
            do i=starts(L+1),starts(L+1)+lengths(L+1)-1
                res(:) = res(:) + coefs(i) * r(xi(1,i),:) * r(xi(2,i),:) * r(xi(3,i),:)
            enddo
        elseif (ll.eq.4) then
            do i=starts(L+1),starts(L+1)+lengths(L+1)-1
                res(:) = res(:) + coefs(i) * r(xi(1,i),:) * r(xi(2,i),:) * r(xi(3,i),:) * r(xi(4,i),:)
            enddo
        elseif (ll.eq.5) then
            do i=starts(L+1),starts(L+1)+lengths(L+1)-1
                res(:) = res(:) + coefs(i) * r(xi(1,i),:) * r(xi(2,i),:) * r(xi(3,i),:) * r(xi(4,i),:) * r(xi(5,i),:)
            enddo
        elseif (ll.eq.6) then
            do i=starts(L+1),starts(L+1)+lengths(L+1)-1
                res(:) = res(:) + coefs(i) * r(xi(1,i),:) * r(xi(2,i),:) * r(xi(3,i),:) * r(xi(4,i),:) * r(xi(5,i),:) * r(xi(6,i),:)
            enddo
        endif
    end subroutine eval_sph_harm_fast

    subroutine recursive_sph_harm_t2( lm_max, r, res, n )
        ! WARNING: No error checking, and assume lm_max is a square
        ! greater than 4
        integer, intent(in) :: lm_max
        real(8), intent(in) :: r(3,n)
        real(8), intent(out) :: res(lm_max,n)
        integer, intent(in) :: n

        integer :: i,l,m,lm,lmax
        real(8) :: fac, rt2, rt3, f0
        complex(8) :: xy

        rt2 = sqrt(2.0d0)
        rt3 = sqrt(3.0d0)
        f0 = 1.0d0 / sqrt(16.0d0 * atan(1.0d0))
        lmax = int(sqrt(dble(lm_max)-1) + 1e-7)

        if (lm_max.le.4) then
            return
        endif

        if (lm_max_buf.ne.lm_max) then
            if (lm_max_buf.ge.0) then
                deallocate( coef0, coef1, c0, c1, ylm )
            endif
            lm_max_buf = lm_max
            allocate ( coef0(lmax+1, lmax+1) )
            allocate ( coef1(lmax+1, lmax+1) )
            allocate ( c0(lmax+1) )
            allocate ( c1(lmax+1) )
            allocate ( ylm(lmax+1, lmax+1) )

            do l=0,lmax
                do m=0,lmax
                    if ((m+2).le.l) then
                        coef0(m+1,l+1) = sqrt( dble( (2*l+3) * (l-m) * (l-m-1) ) / dble( (2*l-1) * (l+m+2) * (l+m+1) ) )
                    else
                        coef0(m+1,l+1) = 0.0d0
                    endif
                    if (m.le.l) then
                        coef1(m+1,l+1) = -sqrt( dble( (2*l+3) * (2*l+1) ) / dble( (l+m+2) * (l+m+1) ) )
                    else
                        coef1(m+1,l+1) = 0.0d0
                    endif
                enddo
                c0(l+1) = sqrt( dble( (2*l+3) * (2*l+1) ) ) / (l+1)
                c1(l+1) = sqrt( dble( (2*l+3) ) / (2*l-1) ) * dble(l) / (l+1)
            enddo
        endif

        do i=1,n
            xy = cmplx(r(1,i), r(2,i), kind=8)
            ylm(:, :) = cmplx(0.0d0, 0.0d0, kind=8)
            ylm(1, 1) = f0
            ylm(1, 2) = rt3 * ylm(1, 1) * r(3, i)
            res(1, i) = f0
            res(3, i) = real( ylm(1, 2) )

            ylm(2, 2) = coef1(1, 1) * xy * ylm(1, 1)
            fac = -rt2
            res(2, i) = fac * aimag( ylm(2, 2) )
            res(4, i) = fac * real( ylm(2, 2) )
            do l=1,lmax-1
                lm = l*l+l+l+l+3
                ylm(1, l+2) = c0(l+1) * r(3, i) * ylm(1, l+1) - c1(l+1) * ylm(1, l)
                res(lm, i) = real(ylm(1, l+2))
                fac = -rt2
                do m=0,l
                    ylm(m+2, l+2) = coef0(m+1,l+1) * ylm(m+2, l) + coef1(m+1,l+1) * xy * ylm(m+1, l+1)
                    res(lm-m-1, i) = fac * aimag( ylm(m+2, l+2) )
                    res(lm+m+1, i) = fac * real( ylm(m+2, l+2) )
                    fac = -fac
                enddo
            enddo
        enddo

    end subroutine recursive_sph_harm_t2

    subroutine recursive_sph_harm_t2_deriv( lm_max, r, res, dres, n )
        ! WARNING: No error checking, and assume lm_max is a square
        ! greater than 4
        integer, intent(in) :: lm_max
        real(8), intent(in) :: r(3,n)
        real(8), intent(out) :: res(lm_max,n)
        real(8), intent(out) :: dres(3,lm_max,n)
        integer, intent(in) :: n

        integer :: i,l,m,lm,lmax
        real(8) :: fac, rt2, rt3, f0
        complex(8) :: xy, ii
        complex(8), allocatable :: dylm(:,:,:)

        rt2 = sqrt(2.0d0)
        rt3 = sqrt(3.0d0)
        f0 = 1.0d0 / sqrt(16.0d0 * atan(1.0d0))
        lmax = int(sqrt(dble(lm_max)-1) + 1e-7)

        if (lm_max.le.4) then
            return
        endif

        if (lm_max_buf.ne.lm_max) then
            if (lm_max_buf.ge.0) then
                deallocate( coef0, coef1, c0, c1, ylm )
            endif
            lm_max_buf = lm_max
            allocate ( coef0(lmax+1, lmax+1) )
            allocate ( coef1(lmax+1, lmax+1) )
            allocate ( c0(lmax+1) )
            allocate ( c1(lmax+1) )
            allocate ( ylm(lmax+1, lmax+1) )

            do l=0,lmax
                do m=0,lmax
                    if ((m+2).le.l) then
                        coef0(m+1,l+1) = sqrt( dble( (2*l+3) * (l-m) * (l-m-1) ) / dble( (2*l-1) * (l+m+2) * (l+m+1) ) )
                    else
                        coef0(m+1,l+1) = 0.0d0
                    endif
                    if (m.le.l) then
                        coef1(m+1,l+1) = -sqrt( dble( (2*l+3) * (2*l+1) ) / dble( (l+m+2) * (l+m+1) ) )
                    else
                        coef1(m+1,l+1) = 0.0d0
                    endif
                enddo
                c0(l+1) = sqrt( dble( (2*l+3) * (2*l+1) ) ) / (l+1)
                c1(l+1) = sqrt( dble( (2*l+3) ) / (2*l-1) ) * dble(l) / (l+1)
            enddo
        endif

        allocate ( dylm(3,lmax+1,lmax+1) )

        ii = cmplx(0.0d0, 1.0d0, kind=8)

        do i=1,n
            xy = cmplx(r(1,i), r(2,i), kind=8)
            ylm(:, :) = cmplx(0.0d0, 0.0d0, kind=8)
            dylm(:, :, :) = cmplx(0.0d0, 0.0d0, kind=8)
            ylm(1, 1) = f0
            res(1, i) = f0
            dres(:, 1, i) = 0.0d0

            ylm(1, 2) = rt3 * ylm(1, 1) * r(3, i)
            dylm(3, 1, 2) = rt3 * ylm(1, 1)
            res(3, i) = real( ylm(1, 2) )
            dres(:, 3, i) = real( dylm(:, 1, 2) )

            ylm(2, 2) = coef1(1, 1) * xy * ylm(1, 1)
            dylm(1, 2, 2) = coef1(1, 1) * ylm(1, 1)
            dylm(2, 2, 2) = coef1(1, 1) * ylm(1, 1) * ii
            fac = -rt2
            res(2, i) = fac * aimag( ylm(2, 2) )
            res(4, i) = fac * real( ylm(2, 2) )
            dres(:, 2, i) = fac * aimag( dylm(:,2,2) )
            dres(:, 4, i) = fac * real( dylm(:,2,2) )
            do l=1,lmax-1
                lm = l*l+l+l+l+3
                ylm(1, l+2) = c0(l+1) * r(3, i) * ylm(1, l+1) - c1(l+1) * ylm(1, l)
                dylm(:, 1, l+2) = c0(l+1) * r(3, i) * dylm(:, 1, l+1) - c1(l+1) * dylm(:, 1, l)
                dylm(3, 1, l+2) = dylm(3, 1, l+2) + c0(l+1) * ylm(1, l+1)

                res(lm, i) = real(ylm(1, l+2))
                dres(:, lm, i) = real(dylm(:, 1, l+2))
                fac = -rt2
                do m=0,l
                    ylm(m+2, l+2) = coef0(m+1,l+1) * ylm(m+2, l) + coef1(m+1,l+1) * xy * ylm(m+1, l+1)
                    dylm(:, m+2, l+2) = coef0(m+1,l+1) * dylm(:,m+2,l) + coef1(m+1,l+1) * xy * dylm(:,m+1,l+1)
                    dylm(1, m+2, l+2) = dylm(1, m+2, l+2) + coef1(m+1,l+1) * ylm(m+1, l+1)
                    dylm(2, m+2, l+2) = dylm(2, m+2, l+2) + ii * coef1(m+1,l+1) * ylm(m+1, l+1)
                    res(lm-m-1, i) = fac * aimag( ylm(m+2, l+2) )
                    res(lm+m+1, i) = fac * real( ylm(m+2, l+2) )
                    dres(:, lm-m-1, i) = fac * aimag( dylm(:, m+2, l+2) )
                    dres(:, lm+m+1, i) = fac * real( dylm(:, m+2, l+2) )
                    fac = -fac
                enddo
            enddo
        enddo

        deallocate( dylm )

    end subroutine recursive_sph_harm_t2_deriv

    subroutine recursive_sph_harm_t2_pyscf( lm_max, r, res, n )
        integer, intent(in) :: lm_max
        real(8), intent(in) :: r(3,n)
        real(8), intent(out) :: res(lm_max,n)
        integer, intent(in) :: n

        call recursive_sph_harm_t2( lm_max, r, res, n )

    end subroutine recursive_sph_harm_t2_pyscf

    subroutine recursive_sph_harm_nt( lm_max, r, res, n )
        integer, intent(in) :: lm_max
        real(8), intent(in) :: r(3,n)
        real(8), intent(out) :: res(n,lm_max)
        integer, intent(in) :: n

        real(8) :: tmp_res(lm_max,n)
        integer :: in, il

        call recursive_sph_harm_t2( lm_max, r, tmp_res, n )
        do il=1,lm_max
            ! res l=1 order: x y z
            ! tmp_res l=1 order: y z x
            ! res(2/x) = tmp_res(4/x)
            ! res(3/x) = tmp_res(2/y)
            ! res(4/z) = tmp_res(3/z)
            if (il.eq.2) then
                do in=1,n
                    res(in,2) = tmp_res(4,in)
                enddo
            elseif (il.eq.3) then
                do in=1,n
                    res(in,3) = tmp_res(2,in)
                enddo
            elseif (il.eq.4) then
                do in=1,n
                    res(in,4) = tmp_res(3,in)
                enddo
            else
                do in=1,n
                    res(in,il) = tmp_res(il,in)
                enddo
            endif
        enddo
    end subroutine recursive_sph_harm_nt


end module fast_sph_harm

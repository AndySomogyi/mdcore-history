subroutine ftest ( x )
    real(8), dimension(:,:), allocatable :: x
    integer :: i, j
    do i=1,10
        do j=1,3
            print *, x(j,i)
        enddo
    enddo
    deallocate(x)
    allocate(x(3,10))
    do i=1,10
        do j=1,3
            x(j,i) = (i-1)*3+j
        enddo
    enddo
end subroutine ftest

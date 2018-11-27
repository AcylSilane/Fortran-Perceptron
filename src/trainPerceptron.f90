! Implementantion of the mark-1 perceptron
! Follows the optimization algorithm presented by Minsky and Papert
! In the algorithm, we:
!   1. Initialize the weights randomly
!   2. Pick a random input
!   3. If classified correctly, do nothing
!   4. If classified incorrectly:
!       4a. If false positive, decrease all weights by x
!       4b. If false negative, increase all weights by x
!   5. If no weights were classified incorrectly, we're done
!   If linearly separable, should converge to the right weights

      subroutine getConsoleInput(x)
        ! Gets an input vector from the console
        integer, dimension(1:784), intent(inout) :: x
        integer :: i
        read(5,*) x(:)
      end subroutine getConsoleInput

      subroutine monochrome(x)
        ! Takes a greyscale input and converts to monochrome
        integer, dimension(1:784), intent(inout) :: x
        integer :: i
        do i = 1, 784, 1
            if (x(i) < 128) then
                x(i) = 0
            else
                x(i) = 1
            endif
        end do
      end subroutine monochrome
      
      function heaviside(x)
        ! Heaviside step function
        ! Returns 1 if x >= 0, and 0 if x < 0
        real, intent(in) :: x
        integer :: heaviside
        if (x >= 0) then
            heaviside = 1
        else
            heaviside = 0
        endif
      end function heaviside
      
      function neuron(input, weights, bias) result(predicted)
          ! A neuron function; multiplies all inputs by weights, sums them,
          ! then adds a bias. Returns true or false based on activation
          ! function.
          integer, dimension(1:784), intent(in) :: input
          real, dimension(1:784), intent(in) :: weights
          real, intent(in) :: bias
          real :: potential
          integer :: predicted
          integer :: heaviside
          ! Multiply all inputs by weights
          potential =  dot_product(input, weights) + bias
          predicted = heaviside(potential)
      end function neuron
      
      program perceptron
        ! Take in a 28 * 28 image = 784 pixels
        implicit none
        integer, dimension(1:784) :: input ! Input vector
        integer, dimension(0:9) :: output ! Output vector
        real, dimension(0:9, 1:784) :: weights ! Weights vector
        real, dimension(0:9) :: bias ! Bias vector
        integer :: truth  ! The true value of the number
        integer :: prediction ! Predicted value of the neuron
        integer :: neuron ! Neuron function output
        integer :: i,j,k    ! Loop counters
        integer :: truePos, falsePos ! True and false positive counters
        
        ! Initialize random seed
        ! This segment of code mostly based on a Linux forum post
        integer :: values(1:8), nonce
        integer, dimension(:), allocatable :: seed
        write(*,*) "Initializing random seed..."
        call date_and_time(VALUES = values)
        call random_seed(size = nonce)
        allocate(seed(1:nonce))
        seed(:) = values(8)
        call random_seed(put=seed)
        
        ! Initialize bias
        write(*,*) "Starting bias at 0..."
        bias = 0
        
        ! Randomize the weights
        write(*,*) "Assigning random weights..."
        do i = 0, 9, 1
            do j = 1, 784, 1
                call random_number(weights(i,j))
            end do
        end do
        
        
        ! Open weights file and bias file
        open(UNIT=12, FILE = "weights.csv", ACTION = "WRITE")
        open(UNIT=13, FILE = "bias.csv", ACTION = "WRITE")

        
        ! Open the training set
        write(*,*) "Acquiring training data..."
        open(UNIT = 10, FILE = "training.csv", FORM = "FORMATTED", STATUS = "OLD", ACTION = "READ")
        write(*,*) "Training..."
        do k = 1, 5, 1 ! Number of training loops
            write(*,*) "Iteration number ", k
            write(*,*) "Training weights..."
            do i = 1, 60000, 1 !MNIST dataset has 60,000 numbers in the training set
                read(10,*) truth, input(:)
                call monochrome(input)
                ! Update Neurons
                do j = 0, 9, 1
                    prediction = neuron(input, weights(j,:), bias(j))
                    if (prediction == 1) then
                        if (j == truth) then
                            ! True positive
                            ! Do nothing; we predicted correctly
                            continue
                        else
                            ! False positive
                            ! Subtract x from our weights
                            weights(j,1:784) = weights(j,1:784) - input(1:784)
                        endif
                    else
                        if (j /= truth) then
                            ! True negative
                            ! Do nothing; we predicted correctly
                            continue
                        else
                            ! False negative
                            ! Add x to all of our weights
                            weights(j, 1:784) = weights(j, 1:784) + input(1:784)
                        endif
                    endif
                end do
            end do
            write(*,*) "Training bias..."
            rewind(10)
            do i=1, 60000, 1
                read(10,*) truth, input(:)
                call monochrome(input)
                ! Update neurons
                do j = 0, 9, 1
                    prediction = neuron(input, weights(j, :), bias(j))
                    if (prediction == 1) then
                        if (j == truth) then
                            ! True positive
                            continue
                        else
                            ! False positive
                            bias(j) = bias(j) - 1
                        endif
                    else
                        if (j /= truth) then
                            ! True negative
                            continue
                        else
                            ! False negative
                            bias(j) = bias(j) + 1
                        endif
                    endif
                end do
            end do
            rewind(10)
        end do
        close(10)
                
        ! Open the test set
        write(*,*) "Acquiring test data..."
        open(UNIT = 11, FILE = "test.csv", FORM = "FORMATTED", STATUS = "OLD", ACTION = "READ")
        truePos = 0 ! True positives counter
        falsePos = 0 ! False positive counter
        do i = 1, 10000, 1 !MNIST dataset has 10,000 numbers in the test set
            read(11, *) truth, input(:)
            call monochrome(input)
            !write(*,*) "Truth: ", truth
            do j = 0, 9, 1
                prediction = neuron(input, weights(j, :), bias(j))
                if (prediction == 1) then
                    !write(*,*) "Prediction: ", j
                    if (j == truth) then
                        truePos = truePos + 1
                    else
                        falsePos = falsePos + 1
                    endif
                endif
            end do
        end do
        close(11)
        write(*,*) "Correct positives: ", truePos
        write(*,*) "False positives: ", falsePos
        
        ! Ready for input
20      write(*,*) "Ready for input. Enter an input vector."
        call getConsoleInput(input)
        do i = 0, 9, 1
            if (neuron(input, weights(i, :), bias(i)) == 1) then
                write(*,*) "I see ", i
            endif
        end do
        
        ! No halting condition; just ctrl+c the process away...
        goto 20

      end program perceptron

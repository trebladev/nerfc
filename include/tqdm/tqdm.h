#ifndef TQDM_H
#define TQDM_H
#include <iostream>
#include <iomanip>
#include <chrono>
#include <iterator>

class tqdm {
public:
    class iterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = int;
        using difference_type = int;
        using pointer = int*;
        using reference = int&;

        iterator(tqdm* parent, int current)
                : parent_(parent), current_(current) {}

        int operator*() const { return current_; }
        iterator& operator++() {
            ++current_;
            parent_->update();
            return *this;
        }

        bool operator!=(const iterator& other) const {
            return current_ != other.current_;
        }

    private:
        tqdm* parent_;
        int current_;
    };

    tqdm(int total_iterations, int report_every = 1)
            : total_iterations_(total_iterations),
              report_every_(report_every),
              start_time_(std::chrono::steady_clock::now()) {}

    iterator begin() {
        return iterator(this, 0);
    }

    iterator end() {
        return iterator(this, total_iterations_);
    }

    void update() {
        ++current_iteration_;
        if (current_iteration_ % report_every_ == 0) {
            display_progress_bar();
        }
    }

private:
    int total_iterations_;
    int report_every_;
    int current_iteration_ = 0;
    std::chrono::steady_clock::time_point start_time_;

    void display_progress_bar() {
        float progress = float(current_iteration_) / float(total_iterations_);
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time_).count();
        float iterations_per_second = current_iteration_ / (elapsed_time / 1000.0);

        int bar_width = 50;
//        std::cout << "[";
        printf("[");
        int position = bar_width * progress;
        for (int i = 0; i < bar_width; ++i) {
//            if (i < position) std::cout << "=";
//            else if (i == position) std::cout << ">";
//            else std::cout << " ";
            if (i < position) printf("=");
            else if (i == position) printf(">");
            else printf(" ");
        }

        printf("] %d%%  %.2f iter/s\r", int(progress * 100.0), iterations_per_second);
        fflush(stdout);
//        std::cout << "] " << int(progress * 100.0) << "%  " << iterations_per_second << " iter/s\r";
//        std::cout.flush();
    }
};
#endif //TQDM_H
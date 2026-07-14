#include "uci/uci.h"
#include <iostream>

int main()
{
  std::cout.setf(std::ios::unitbuf); // flush every line — required for UCI GUIs
  uci::loop();
  return 0;
}

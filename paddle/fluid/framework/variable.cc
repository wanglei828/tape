//  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "paddle/fluid/framework/variable.h"

namespace paddle {
namespace fluid {
namespace framework {

bool Variable::IsInitialized() const { return holder_ != nullptr; }

void Variable::Clear() { holder_.reset(); }

std::type_index Variable::Type() const {
  PADDLE_ENFORCE(holder_ != nullptr, "Must hold memory");
  return holder_->Type();
}

}  // namespace framework
}  // namespace fluid
}  // namespace paddle
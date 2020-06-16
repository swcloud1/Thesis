"""
/*
 * Copyright 2020 Bloomreach B.V. (http://www.bloomreach.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 """
 
from enum import Enum

class AdjustmentType(Enum):
    MORE_INTENSE_MAIN = 1
    LESS_INTENSE_MAIN = 2
    MORE_INTENSE_SPECIFIC = 3
    LESS_INTENSE_SPECIFIC = 4
    REPLACE = 5
    REMOVE = 6

    FOCUS_SR = 7
    FOCUS_EM = 8

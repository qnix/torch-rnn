require 'torch'
require 'nn'
require 'optim'

require 'LanguageModel'
require 'util.DataLoader'

local utils = require 'util.utils'
local unpack = unpack or table.unpack

local cmd = torch.CmdLine()

-- Dataset options
cmd:option('-input_h5', 'data/tiny-shakespeare.h5')
cmd:option('-input_json', 'data/tiny-shakespeare.json')
cmd:option('-batch_size', 50)
cmd:option('-seq_length', 50)

-- Model options
cmd:option('-init_from', '')
cmd:option('-reset_iterations', 1)
cmd:option('-model_type', 'lstm')
cmd:option('-wordvec_size', 64)
cmd:option('-rnn_size', 128)
cmd:option('-num_layers', 2)
cmd:option('-dropout', 0)
cmd:option('-batchnorm', 0)

-- Optimization options
cmd:option('-max_epochs', 100)
cmd:option('-lr', 1.25e-3)
cmd:option('-min_lr', 2.5e-4)
cmd:option('-grad_clip', 5)

-- Output options
cmd:option('-print_every', 50)
cmd:option('-checkpoint_every', 0)
cmd:option('-checkpoint_name', 'cv/checkpoint')

-- Benchmark options
cmd:option('-speed_benchmark', 0)
cmd:option('-memory_benchmark', 0)

-- Backend options
cmd:option('-gpu', 0)
cmd:option('-gpu_backend', 'cuda')

local opt = cmd:parse(arg)


-- Set up GPU stuff
local dtype = 'torch.FloatTensor'
if opt.gpu >= 0 and opt.gpu_backend == 'cuda' then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpu + 1)
  dtype = 'torch.CudaTensor'
  print(string.format('Running with CUDA on GPU %d', opt.gpu))
elseif opt.gpu >= 0 and opt.gpu_backend == 'opencl' then
  -- Memory benchmarking is only supported in CUDA mode
  -- TODO: Time benchmarking is probably wrong in OpenCL mode.
  require 'cltorch'
  require 'clnn'
  cltorch.setDevice(opt.gpu + 1)
  dtype = torch.Tensor():cl():type()
  print(string.format('Running with OpenCL on GPU %d', opt.gpu))
else
  -- Memory benchmarking is only supported in CUDA mode
  opt.memory_benchmark = 0
  print 'Running in CPU mode'
end


-- Initialize the DataLoader and vocabulary
local loader = DataLoader(opt)
local vocab = utils.read_json(opt.input_json)
local idx_to_token = {}
for k, v in pairs(vocab.idx_to_token) do
  idx_to_token[tonumber(k)] = v
end

-- Initialize the model and criterion
local opt_clone = torch.deserialize(torch.serialize(opt))
opt_clone.idx_to_token = idx_to_token
local model = nil
local start_i = 0
if opt.init_from ~= '' then
  print('Initializing from ', opt.init_from)
  local checkpoint = torch.load(opt.init_from)
  model = checkpoint.model:type(dtype)
  if opt.reset_iterations == 0 then
    start_i = checkpoint.i
  end
else
  model = nn.LanguageModel(opt_clone):type(dtype)
end
local params, grad_params = model:getParameters()
local crit = nn.CrossEntropyCriterion():type(dtype)

-- Set up some variables we will use below
local N, T = opt.batch_size, opt.seq_length
local train_loss_history = {}
local val_loss_history = {}
local val_loss_history_it = {}
local forward_backward_times = {}
local init_memory_usage, memory_usage = nil, {}

if opt.memory_benchmark == 1 then
  -- This should only be enabled in GPU mode
  assert(cutorch)
  cutorch.synchronize()
  local free, total = cutorch.getMemoryUsage(cutorch.getDevice())
  init_memory_usage = total - free
end

-- Loss function that we pass to an optim method
local function f(w)
  assert(w == params)
  grad_params:zero()

  -- Get a minibatch and run the model forward, maybe timing it
  local timer
  local x, y = loader:nextBatch('train')
  x, y = x:type(dtype), y:type(dtype)
  if opt.speed_benchmark == 1 then
    if cutorch then cutorch.synchronize() end
    timer = torch.Timer()
  end
  local scores = model:forward(x)

  -- Use the Criterion to compute loss; we need to reshape the scores to be
  -- two-dimensional before doing so. Annoying.
  local scores_view = scores:view(N * T, -1)
  local y_view = y:view(N * T)
  local loss = crit:forward(scores_view, y_view)

  -- Run the Criterion and model backward to compute gradients, maybe timing it
  local grad_scores = crit:backward(scores_view, y_view):view(N, T, -1)
  model:backward(x, grad_scores)
  if timer then
    if cutorch then cutorch.synchronize() end
    local time = timer:time().real
    print('Forward / Backward pass took ', time)
    table.insert(forward_backward_times, time)
  end

  -- Maybe record memory usage
  if opt.memory_benchmark == 1 then
    assert(cutorch)
    if cutorch then cutorch.synchronize() end
    local free, total = cutorch.getMemoryUsage(cutorch.getDevice())
    local memory_used = total - free - init_memory_usage
    local memory_used_mb = memory_used / 1024 / 1024
    print(string.format('Using %dMB of memory', memory_used_mb))
    table.insert(memory_usage, memory_used)
  end

  if opt.grad_clip > 0 then
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  end

  return loss, grad_params
end

function make_avg_loss_check(period)
   local env = {}
   local idx = 1
   env.last_period = {}
   env.current_period = {}
   function env.accumulate(loss)
      if env.current_period[idx] ~= nil then
         env.last_period[idx] = env.current_period[idx]
      end
      env.current_period[idx] = loss
      idx = (idx + 1) % period + 1
   end
   function average(losses)
      local sum = 0
      for i = 1, #losses do
         sum = sum + losses[i]
      end
      return (sum / #losses)
   end
   function env.current_average()
      return average(env.current_period)
   end
   function env.last_average()
      return average(env.last_period)
   end
   function env.is_smaller()
      return env.current_average() < env.last_average()
   end
   return env
end

function make_train_loss(period)
   local train_loss = {}
   function accumulate(iteration, loss)
      local idx = (iteration % period) + 1
      train_loss[idx] = loss
   end
   function average()
      local sum = 0
      for i = 1, #train_loss do
         sum = sum + train_loss[i]
      end
      return (sum / #train_loss)
   end
   return accumulate, average
end

function make_time_tracker()
   local timer = torch.Timer()
   local start_time = timer:time().real
   function elapse(with_reset)
      local current_time = timer:time().real
      local elapsed_time = current_time - start_time
      if with_reset ~= nil then
         start_time = timer:time().real
      end
      return elapsed_time
   end
   return elapse
end

-- Train the model!
local optim_config = {learningRate = opt.lr}
local num_train = loader.split_sizes['train']
local num_iterations = opt.max_epochs * num_train
local acc_loss, avg_loss = make_train_loss(10)
local avg_loss_chk = make_avg_loss_check(50)
local elapse_timer = make_time_tracker()

model:training()
for i = start_i + 1, num_iterations do
  local epoch = math.floor(i / num_train) + 1

  -- Check if we are at the end of an epoch
  if i % num_train == 0 then
    model:resetStates() -- Reset hidden states

    -- Maybe decay learning rate
    optim_config = { learningRate = opt.min_lr + (opt.lr - opt.min_lr) * math.exp(-i/num_iterations) }
  end

  if (i % opt.print_every) == 0 and not avg_loss_chk.is_smaller() then
     local old_lr = optim_config.learningRate
     optim_config = { learningRate = opt.min_lr + (old_lr - opt.min_lr) * math.exp(-i/num_iterations) }
  end

  -- Take a gradient step and maybe print
  -- Note that adam returns a singleton array of losses
  local _, loss = optim.adam(f, params, optim_config)
  table.insert(train_loss_history, loss[1])
  if opt.print_every > 0 and i % opt.print_every == 0 then
    local float_epoch = i / num_train + 1
    acc_loss(i, loss[1])
    local msg = 'Epoch %.2f / %d, i = %d / %d, loss = %f, lrate: %f, time: %f'
    local args = {msg, float_epoch, opt.max_epochs, i, num_iterations, avg_loss(), optim_config.learningRate, elapse_timer(true) }
    print(string.format(unpack(args)))
  end

  -- Maybe save a checkpoint
  local check_every = opt.checkpoint_every
  if (check_every > 0 and i % check_every == 0) or i % num_train == 0 or i == num_iterations then
    -- Evaluate loss on the validation set. Note that we reset the state of
    -- the model; this might happen in the middle of an epoch, but that
    -- shouldn't cause too much trouble.
    model:evaluate()
    model:resetStates()
    local num_val = loader.split_sizes['val']
    local val_loss = 0
    for j = 1, num_val do
      local xv, yv = loader:nextBatch('val')
      xv = xv:type(dtype)
      yv = yv:type(dtype):view(N * T)
      local scores = model:forward(xv):view(N * T, -1)
      val_loss = val_loss + crit:forward(scores, yv)
    end
    val_loss = val_loss / num_val
    print('val_loss = ', val_loss, ', val_time = ', elapse_timer(true))
    table.insert(val_loss_history, val_loss)
    table.insert(val_loss_history_it, i)
    model:resetStates()
    model:training()

    -- First save a JSON checkpoint, excluding the model
    local checkpoint = {
      opt = opt,
      train_loss_history = train_loss_history,
      val_loss_history = val_loss_history,
      val_loss_history_it = val_loss_history_it,
      forward_backward_times = forward_backward_times,
      memory_usage = memory_usage,
      i = i
    }
    local filename = string.format('%s_%d.json', opt.checkpoint_name, i)
    -- Make sure the output directory exists before we try to write it
    paths.mkdir(paths.dirname(filename))
    utils.write_json(filename, checkpoint)

    -- Now save a torch checkpoint with the model
    -- Cast the model to float before saving so it can be used on CPU
    model:clearState()
    model:float()
    checkpoint.model = model
    local filename = string.format('%s_%d_%0.6f_%0.6f.t7', opt.checkpoint_name, i, avg_loss(), val_loss)
    paths.mkdir(paths.dirname(filename))
    torch.save(filename, checkpoint)
    model:type(dtype)
    params, grad_params = model:getParameters()
    collectgarbage()

    -- Reset the timer
    elapse_timer(true)
  end
end

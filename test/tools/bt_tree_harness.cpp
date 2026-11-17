// A test harness for the task_program trees TSA generates (vNext M4.5). Built and run by
// test/test_m45_composition.py; never shipped.
//
//   --parse TREE.xml          (built with -DWITH_TMR, linked to TMR's library) build the tree with TMR's OWN node
//                             palette (trainit::registerAllNodes: every node name and port as the runtime declares
//                             it): the runtime's factory accepts the tree.
//   --run TREE.xml SCENARIO   RUN the tree in BehaviorTree.CPP with every non-builtin node replaced by a scripted
//                             stub (its ports: the attributes the tree gives it; `fault`, `reason`, `subject` are
//                             outputs). The standard nodes -- Sequence, Fallback, RetryUntilSuccessful, Repeat,
//                             Script, AlwaysFailure, _failureIf, _skipIf, SubTree _autoremap -- are the real ones.
//
// SCENARIO, one line per scripted outcome, consumed in order per node identity (an identity with no line left
// ticks SUCCESS):
//   <identity> -> SUCCESS | FAILURE [port=value ...] [xN]
// <identity> = the node's name, then ':' and its first present attribute among waypoint, detector, expect, index
// (MoveWaypoint:bin_view, VerifyGrasp:held, SetWaypointFromCandidate:1, CloseGripper). An output is written only
// when the tree remaps it, as the runtime's nodes do. The trace: "TICK <identity> SUCCESS|FAILURE",
// "LOG <message>" (a Log node, its message read through the blackboard), then "RESULT SUCCESS|FAILURE".
#include <behaviortree_cpp/bt_factory.h>

#include <algorithm>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#ifdef WITH_TMR
namespace trainit
{
void registerAllNodes(BT::BehaviorTreeFactory& factory);
}
#endif

namespace
{
struct Outcome
{
  BT::NodeStatus status{BT::NodeStatus::SUCCESS};
  std::vector<std::pair<std::string, std::string>> outputs;
};

std::map<std::string, std::deque<Outcome>> g_script;
long g_ticks = 0;
const std::set<std::string> kOutputs{"fault", "reason", "subject"};

class Stub : public BT::SyncActionNode
{
public:
  Stub(const std::string& name, const BT::NodeConfig& config, std::string id)
    : BT::SyncActionNode(name, config), id_(std::move(id)) {}

  BT::NodeStatus tick() override
  {
    if (++g_ticks > 100000) throw std::runtime_error("ABORT: more than 100000 ticks");
    if (id_ == "Log")
    {
      auto m = getInput<std::string>("message");
      std::cout << "LOG " << (m ? m.value() : std::string("<unreadable: ") + m.error() + ">") << std::endl;
      return BT::NodeStatus::SUCCESS;
    }
    std::string ident = id_;
    for (const char* key : {"waypoint", "detector", "expect", "index"})
    {
      const auto it = config().input_ports.find(key);
      if (it != config().input_ports.end() && !it->second.empty()) { ident += ":" + it->second; break; }
    }
    Outcome o;
    auto q = g_script.find(ident);
    if (q != g_script.end() && !q->second.empty())
    {
      o = q->second.front();
      q->second.pop_front();
    }
    for (const auto& [port, value] : o.outputs)
    {
      const auto it = config().output_ports.find(port);
      if (it != config().output_ports.end() && !it->second.empty()) setOutput(port, value);
    }
    // TSA 8.8.0 (M4.5d): a motion stub follows TMR 2.9.0's `holding` rule -- a failure while it reads true writes
    // the fault "MOTION_FAULT: <scripted cause> while holding" (TMR's own test pins the node; this pins the TREE).
    // A scripted outcome that carries its own `fault` is written AS SCRIPTED (no suffix): TMR 2.9.0 suffixes every
    // fault written while held -- that rewrite is pinned by TMR's test_m45d_held_motion_fault (HELD-03 / HELD-14),
    // not here; such a test asserts the tree's gating only.
    std::string held;
    if ((id_ == "MoveWaypoint" || id_ == "FollowPath") && config().input_ports.count("holding"))
    {
      const auto h = getInput<bool>("holding");
      held = h ? (h.value() ? " held" : " free") : " held?";   // unreadable = held (TMR fails closed)
      const bool scripted_fault = std::any_of(o.outputs.begin(), o.outputs.end(),
                                              [](const auto& kv) { return kv.first == "fault"; });
      if (o.status == BT::NodeStatus::FAILURE && held != " free" && !scripted_fault)
      {
        const auto it = config().output_ports.find("fault");
        if (it != config().output_ports.end() && !it->second.empty())
          setOutput("fault", std::string("MOTION_FAULT: SCRIPTED during plan while holding"));
      }
    }
    if (!held.empty()) std::cout << "HOLD " << ident << held << std::endl;   // before its TICK line
    std::cout << "TICK " << ident << " " << (o.status == BT::NodeStatus::SUCCESS ? "SUCCESS" : "FAILURE")
              << std::endl;
    return o.status;
  }

private:
  std::string id_;
};

void readScenario(const std::string& path)
{
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line))
  {
    if (line.empty() || line[0] == '#') continue;
    const auto arrow = line.find(" -> ");
    if (arrow == std::string::npos) throw std::runtime_error("bad scenario line: " + line);
    const std::string ident = line.substr(0, arrow);
    std::istringstream rest(line.substr(arrow + 4));
    std::string word;
    Outcome o;
    int times = 1;
    rest >> word;
    o.status = word == "FAILURE" ? BT::NodeStatus::FAILURE : BT::NodeStatus::SUCCESS;
    while (rest >> word)
    {
      const auto eq = word.find('=');
      if (eq == std::string::npos && word.size() > 1 && word[0] == 'x') { times = std::stoi(word.substr(1)); continue; }
      if (eq == std::string::npos) throw std::runtime_error("bad output: " + word);
      o.outputs.emplace_back(word.substr(0, eq), word.substr(eq + 1));
    }
    for (int i = 0; i < times; ++i) g_script[ident].push_back(o);
  }
}

// Every element of the tree that is not a builtin node: a stub with the attributes it is given as its ports.
void registerStubs(BT::BehaviorTreeFactory& factory, const std::string& xml)
{
  static const std::set<std::string> kNotNodes{"root", "BehaviorTree", "SubTree", "TreeNodesModel"};
  std::map<std::string, std::set<std::string>> ports;
  const std::regex element("<([A-Za-z_][A-Za-z0-9_]*)([^>]*)>"), attribute("([A-Za-z_][A-Za-z0-9_]*)=\"");
  for (std::sregex_iterator it(xml.begin(), xml.end(), element), end; it != end; ++it)
  {
    const std::string tag = (*it)[1];
    if (kNotNodes.count(tag) || factory.builtinNodes().count(tag)) continue;
    auto& own = ports[tag];
    const std::string attrs = (*it)[2];
    for (std::sregex_iterator a(attrs.begin(), attrs.end(), attribute), aend; a != aend; ++a)
    {
      const std::string name = (*a)[1];
      if (name != "name" && name[0] != '_') own.insert(name);
    }
  }
  for (const auto& [tag, names] : ports)
  {
    BT::PortsList list;
    for (const auto& n : names)
      list.insert(kOutputs.count(n) ? BT::OutputPort<std::string>(n)
                                    : (n == "holding" ? BT::InputPort<bool>(n) : BT::InputPort<std::string>(n)));
    BT::TreeNodeManifest manifest{BT::NodeType::ACTION, tag, list, {}};
    const std::string id = tag;
    factory.registerBuilder(manifest, [id](const std::string& n, const BT::NodeConfig& c) {
      return std::make_unique<Stub>(n, c, id);
    });
  }
}
}  // namespace

int main(int argc, char** argv)
{
  const std::string mode = argc > 1 ? argv[1] : "";
  if (argc < 3 || (mode != "--parse" && mode != "--run") || (mode == "--run" && argc < 4))
  {
    std::cerr << "usage: bt_tree_harness --parse TREE.xml | --run TREE.xml SCENARIO" << std::endl;
    return 2;
  }
  std::ifstream xml_in(argv[2]);
  std::stringstream xml_text;
  xml_text << xml_in.rdbuf();
  const std::string xml = xml_text.str();
  try
  {
    if (mode == "--parse")
    {
#ifdef WITH_TMR
      BT::BehaviorTreeFactory runtime;
      trainit::registerAllNodes(runtime);
      auto tree = runtime.createTreeFromText(xml);
      std::cout << "PARSED " << tree.subtrees.size() << " trees" << std::endl;
      return 0;
#else
      std::cout << "ERROR built without TMR (-DWITH_TMR)" << std::endl;
      return 1;
#endif
    }
    readScenario(argv[3]);
    BT::BehaviorTreeFactory stubs;
    registerStubs(stubs, xml);
    auto tree = stubs.createTreeFromText(xml);
    const auto status = tree.tickWhileRunning();
    std::cout << "RESULT " << (status == BT::NodeStatus::SUCCESS ? "SUCCESS" : "FAILURE") << std::endl;
    return 0;
  }
  catch (const std::exception& e)
  {
    std::cout << "ERROR " << e.what() << std::endl;
    return 1;
  }
}

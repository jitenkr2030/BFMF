import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Check, Star, Zap, Shield, Globe, Users, Code, Database, Cpu, Heart, BookOpen, Building2, TrendingUp, Microscope, Cloud, Factory } from "lucide-react"

export default function Home() {
  return (
    <div className="flex flex-col items-center min-h-screen p-4 bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <div className="w-full max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="text-center space-y-4 py-8">
          <div className="relative w-24 h-24 md:w-32 md:h-32 mx-auto">
            <img
              src="/bharat-logo.svg"
              alt="Bharat Foundation Model Framework Logo"
              className="w-full h-full object-contain"
            />
          </div>
          <h1 className="text-4xl md:text-6xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            Bharat Foundation Model Framework
          </h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            India's sovereign AI framework powering digital transformation across 10 critical domains
          </p>
        </div>

        {/* Executive Summary */}
        <Card className="border-2 border-blue-200 dark:border-blue-800">
          <CardHeader>
            <CardTitle className="text-2xl flex items-center gap-2">
              <span className="text-blue-600">Framework Overview</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-lg leading-relaxed">
              The Bharat Foundation Model Framework (BFMF) is India's first comprehensive open-source AI framework designed specifically for India's unique digital transformation needs. Built with sovereignty at its core, BFMF enables organizations to develop and deploy AI applications that are truly Indian in context, language, and infrastructure.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-6">
              <div className="text-center p-4 bg-blue-50 dark:bg-blue-950 rounded-lg">
                <div className="text-3xl font-bold text-blue-600">10</div>
                <div className="text-sm text-muted-foreground">Domain Modules</div>
              </div>
              <div className="text-center p-4 bg-green-50 dark:bg-green-950 rounded-lg">
                <div className="text-3xl font-bold text-green-600">22+</div>
                <div className="text-sm text-muted-foreground">Indian Languages</div>
              </div>
              <div className="text-center p-4 bg-purple-50 dark:bg-purple-950 rounded-lg">
                <div className="text-3xl font-bold text-purple-600">100%</div>
                <div className="text-sm text-muted-foreground">Self-Hosted</div>
              </div>
              <div className="text-center p-4 bg-orange-50 dark:bg-orange-950 rounded-lg">
                <div className="text-3xl font-bold text-orange-600">1-Click</div>
                <div className="text-sm text-muted-foreground">Deployment</div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Core Capabilities */}
        <Tabs defaultValue="overview" className="w-full">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="domains">Domains</TabsTrigger>
            <TabsTrigger value="features">Features</TabsTrigger>
            <TabsTrigger value="usecases">Use Cases</TabsTrigger>
            <TabsTrigger value="technical">Technical</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-2xl">Core Capabilities</CardTitle>
                <CardDescription>What makes BFMF India's most powerful AI framework</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  <div className="space-y-3 p-4 bg-blue-50 dark:bg-blue-950 rounded-lg">
                    <div className="flex items-center gap-2">
                      <Globe className="text-blue-600" size={24} />
                      <h3 className="text-lg font-semibold">Multi-Language AI</h3>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Native support for 22+ Indian languages with code-switching capabilities and cultural context understanding.
                    </p>
                    <div className="flex flex-wrap gap-2 mt-2">
                      <Badge variant="secondary">Hindi</Badge>
                      <Badge variant="secondary">Bengali</Badge>
                      <Badge variant="secondary">Tamil</Badge>
                      <Badge variant="secondary">+19 more</Badge>
                    </div>
                  </div>

                  <div className="space-y-3 p-4 bg-green-50 dark:bg-green-950 rounded-lg">
                    <div className="flex items-center gap-2">
                      <Shield className="text-green-600" size={24} />
                      <h3 className="text-lg font-semibold">Sovereign Infrastructure</h3>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Complete data sovereignty with 100% self-hosted infrastructure compliant with Indian regulations.
                    </p>
                    <div className="flex flex-wrap gap-2 mt-2">
                      <Badge variant="secondary">Data Residency</Badge>
                      <Badge variant="secondary">Federated</Badge>
                      <Badge variant="secondary">Compliant</Badge>
                    </div>
                  </div>

                  <div className="space-y-3 p-4 bg-purple-50 dark:bg-purple-950 rounded-lg">
                    <div className="flex items-center gap-2">
                      <Zap className="text-purple-600" size={24} />
                      <h3 className="text-lg font-semibold">Production Ready</h3>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Built for production from day one with one-command deployment, monitoring, and scaling capabilities.
                    </p>
                    <div className="flex flex-wrap gap-2 mt-2">
                      <Badge variant="secondary">1-Click Deploy</Badge>
                      <Badge variant="secondary">Auto-Scaling</Badge>
                      <Badge variant="secondary">Monitoring</Badge>
                    </div>
                  </div>

                  <div className="space-y-3 p-4 bg-orange-50 dark:bg-orange-950 rounded-lg">
                    <div className="flex items-center gap-2">
                      <Code className="text-orange-600" size={24} />
                      <h3 className="text-lg font-semibold">CLI Automation</h3>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Comprehensive command-line interface for workflow automation and streamlined development.
                    </p>
                    <div className="flex flex-wrap gap-2 mt-2">
                      <Badge variant="secondary">Training</Badge>
                      <Badge variant="secondary">Deployment</Badge>
                      <Badge variant="secondary">Management</Badge>
                    </div>
                  </div>

                  <div className="space-y-3 p-4 bg-red-50 dark:bg-red-950 rounded-lg">
                    <div className="flex items-center gap-2">
                      <Database className="text-red-600" size={24} />
                      <h3 className="text-lg font-semibold">Domain-Specific</h3>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      10 specialized modules tailored for India's unique requirements across critical sectors.
                    </p>
                    <div className="flex flex-wrap gap-2 mt-2">
                      <Badge variant="secondary">Governance</Badge>
                      <Badge variant="secondary">Education</Badge>
                      <Badge variant="secondary">Healthcare</Badge>
                    </div>
                  </div>

                  <div className="space-y-3 p-4 bg-cyan-50 dark:bg-cyan-950 rounded-lg">
                    <div className="flex items-center gap-2">
                      <Users className="text-cyan-600" size={24} />
                      <h3 className="text-lg font-semibold">Open Ecosystem</h3>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Open-source framework with growing Indian developer community and comprehensive support.
                    </p>
                    <div className="flex flex-wrap gap-2 mt-2">
                      <Badge variant="secondary">Open Source</Badge>
                      <Badge variant="secondary">Community</Badge>
                      <Badge variant="secondary">Documentation</Badge>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="domains" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-2xl">10 Domain-Specific Modules</CardTitle>
                <CardDescription>Specialized AI capabilities for India's critical sectors</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div className="p-4 border-l-4 border-blue-500 bg-blue-50 dark:bg-blue-950 rounded-r-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <Globe className="text-blue-600" size={20} />
                        <h3 className="text-lg font-semibold">🗣️ Language AI</h3>
                      </div>
                      <p className="text-sm text-muted-foreground mb-2">
                        Multi-language processing with native support for 22+ Indian languages, code-switching, and cultural context.
                      </p>
                      <div className="flex flex-wrap gap-1">
                        <Badge variant="outline">BharatLang</Badge>
                        <Badge variant="outline">Tokenization</Badge>
                        <Badge variant="outline">Translation</Badge>
                      </div>
                    </div>

                    <div className="p-4 border-l-4 border-green-500 bg-green-50 dark:bg-green-950 rounded-r-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <Building2 className="text-green-600" size={20} />
                        <h3 className="text-lg font-semibold">🏛️ Digital Governance AI</h3>
                      </div>
                      <p className="text-sm text-muted-foreground mb-2">
                        Policy analysis, RTI automation, compliance auditing, and government scheme intelligence.
                      </p>
                      <div className="flex flex-wrap gap-1">
                        <Badge variant="outline">BharatGov</Badge>
                        <Badge variant="outline">RTI Assistant</Badge>
                        <Badge variant="outline">Policy Analysis</Badge>
                      </div>
                    </div>

                    <div className="p-4 border-l-4 border-purple-500 bg-purple-50 dark:bg-purple-950 rounded-r-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <BookOpen className="text-purple-600" size={20} />
                        <h3 className="text-lg font-semibold">🎓 Education AI</h3>
                      </div>
                      <p className="text-sm text-muted-foreground mb-2">
                        AI-powered tutoring, content generation, personalized learning, and digital classroom management.
                      </p>
                      <div className="flex flex-wrap gap-1">
                        <Badge variant="outline">BharatEdu</Badge>
                        <Badge variant="outline">AI Teacher</Badge>
                        <Badge variant="outline">Content Gen</Badge>
                      </div>
                    </div>

                    <div className="p-4 border-l-4 border-yellow-500 bg-yellow-50 dark:bg-yellow-950 rounded-r-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <TrendingUp className="text-yellow-600" size={20} />
                        <h3 className="text-lg font-semibold">💰 Financial AI</h3>
                      </div>
                      <p className="text-sm text-muted-foreground mb-2">
                        Financial analysis, transaction auditing, fraud detection, and market prediction for Indian markets.
                      </p>
                      <div className="flex flex-wrap gap-1">
                        <Badge variant="outline">BharatFinGPT</Badge>
                        <Badge variant="outline">AuditGPT</Badge>
                        <Badge variant="outline">Risk Analysis</Badge>
                      </div>
                    </div>

                    <div className="p-4 border-l-4 border-red-500 bg-red-50 dark:bg-red-950 rounded-r-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <Heart className="text-red-600" size={20} />
                        <h3 className="text-lg font-semibold">🏥 Healthcare AI</h3>
                      </div>
                      <p className="text-sm text-muted-foreground mb-2">
                        Telemedicine support, health monitoring, diagnostic assistance, and healthcare management.
                      </p>
                      <div className="flex flex-wrap gap-1">
                        <Badge variant="outline">BharatHealth</Badge>
                        <Badge variant="outline">Telemedicine</Badge>
                        <Badge variant="outline">Diagnostics</Badge>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="p-4 border-l-4 border-orange-500 bg-orange-50 dark:bg-orange-950 rounded-r-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <Factory className="text-orange-600" size={20} />
                        <h3 className="text-lg font-semibold">🌾 Agriculture AI</h3>
                      </div>
                      <p className="text-sm text-muted-foreground mb-2">
                        Crop advisory, yield prediction, pest detection, and agricultural market intelligence.
                      </p>
                      <div className="flex flex-wrap gap-1">
                        <Badge variant="outline">BharatKrishi</Badge>
                        <Badge variant="outline">Crop Advisor</Badge>
                        <Badge variant="outline">Yield Prediction</Badge>
                      </div>
                    </div>

                    <div className="p-4 border-l-4 border-cyan-500 bg-cyan-50 dark:bg-cyan-950 rounded-r-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <Users className="text-cyan-600" size={20} />
                        <h3 className="text-lg font-semibold">📰 Media AI</h3>
                      </div>
                      <p className="text-sm text-muted-foreground mb-2">
                        Content generation, fact-checking, sentiment analysis, and media monitoring for Indian context.
                      </p>
                      <div className="flex flex-wrap gap-1">
                        <Badge variant="outline">BharatMedia</Badge>
                        <Badge variant="outline">Fact Check</Badge>
                        <Badge variant="outline">Content Gen</Badge>
                      </div>
                    </div>

                    <div className="p-4 border-l-4 border-indigo-500 bg-indigo-50 dark:bg-indigo-950 rounded-r-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <Microscope className="text-indigo-600" size={20} />
                        <h3 className="text-lg font-semibold">🧩 Research AI</h3>
                      </div>
                      <p className="text-sm text-muted-foreground mb-2">
                        Academic research tools, dataset creation, literature analysis, and research assistance.
                      </p>
                      <div className="flex flex-wrap gap-1">
                        <Badge variant="outline">BharatResearch</Badge>
                        <Badge variant="outline">Dataset Tools</Badge>
                        <Badge variant="outline">Literature Analysis</Badge>
                      </div>
                    </div>

                    <div className="p-4 border-l-4 border-pink-500 bg-pink-50 dark:bg-pink-950 rounded-r-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <Cloud className="text-pink-600" size={20} />
                        <h3 className="text-lg font-semibold">🔐 Infrastructure AI</h3>
                      </div>
                      <p className="text-sm text-muted-foreground mb-2">
                        Sovereign cloud management, computing federation, and infrastructure optimization.
                      </p>
                      <div className="flex flex-wrap gap-1">
                        <Badge variant="outline">BharatCloud</Badge>
                        <Badge variant="outline">Federation</Badge>
                        <Badge variant="outline">Resource Opt</Badge>
                      </div>
                    </div>

                    <div className="p-4 border-l-4 border-teal-500 bg-teal-50 dark:bg-teal-950 rounded-r-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <Cpu className="text-teal-600" size={20} />
                        <h3 className="text-lg font-semibold">🧠 Enterprise AI</h3>
                      </div>
                      <p className="text-sm text-muted-foreground mb-2">
                        Business applications, private deployment, and enterprise-grade AI solutions.
                      </p>
                      <div className="flex flex-wrap gap-1">
                        <Badge variant="outline">BharatBiz</Badge>
                        <Badge variant="outline">Private Deploy</Badge>
                        <Badge variant="outline">Enterprise Grade</Badge>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="features" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-2xl">Key Features & Functions</CardTitle>
                <CardDescription>Comprehensive capabilities of the BFMF framework</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-4">
                      <h3 className="text-xl font-semibold text-blue-600">🚀 Core Framework Features</h3>
                      <div className="space-y-3">
                        <div className="flex items-start gap-3">
                          <Check className="text-green-500 mt-1" size={20} />
                          <div>
                            <h4 className="font-semibold">Modular Architecture</h4>
                            <p className="text-sm text-muted-foreground">Each domain module works independently or combines seamlessly</p>
                          </div>
                        </div>
                        <div className="flex items-start gap-3">
                          <Check className="text-green-500 mt-1" size={20} />
                          <div>
                            <h4 className="font-semibold">Unified CLI</h4>
                            <p className="text-sm text-muted-foreground">Single command-line interface for all operations</p>
                          </div>
                        </div>
                        <div className="flex items-start gap-3">
                          <Check className="text-green-500 mt-1" size={20} />
                          <div>
                            <h4 className="font-semibold">Model Registry</h4>
                            <p className="text-sm text-muted-foreground">Integrated model management and versioning</p>
                          </div>
                        </div>
                        <div className="flex items-start gap-3">
                          <Check className="text-green-500 mt-1" size={20} />
                          <div>
                            <h4 className="font-semibold">Evaluation Framework</h4>
                            <p className="text-sm text-muted-foreground">Comprehensive benchmarking and evaluation tools</p>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-4">
                      <h3 className="text-xl font-semibold text-green-600">🛠️ Development Features</h3>
                      <div className="space-y-3">
                        <div className="flex items-start gap-3">
                          <Check className="text-green-500 mt-1" size={20} />
                          <div>
                            <h4 className="font-semibold">Pre-trained Models</h4>
                            <p className="text-sm text-muted-foreground">Domain-specific models ready for fine-tuning</p>
                          </div>
                        </div>
                        <div className="flex items-start gap-3">
                          <Check className="text-green-500 mt-1" size={20} />
                          <div>
                            <h4 className="font-semibold">Fine-tuning Tools</h4>
                            <p className="text-sm text-muted-foreground">Specialized fine-tuning for Indian contexts</p>
                          </div>
                        </div>
                        <div className="flex items-start gap-3">
                          <Check className="text-green-500 mt-1" size={20} />
                          <div>
                            <h4 className="font-semibold">Dataset Management</h4>
                            <p className="text-sm text-muted-foreground">Curated datasets for Indian use cases</p>
                          </div>
                        </div>
                        <div className="flex items-start gap-3">
                          <Check className="text-green-500 mt-1" size={20} />
                          <div>
                            <h4 className="font-semibold">API Integration</h4>
                            <p className="text-sm text-muted-foreground">Seamless integration with existing systems</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-4">
                      <h3 className="text-xl font-semibold text-purple-600">🔒 Security & Compliance</h3>
                      <div className="space-y-3">
                        <div className="flex items-start gap-3">
                          <Check className="text-green-500 mt-1" size={20} />
                          <div>
                            <h4 className="font-semibold">Data Sovereignty</h4>
                            <p className="text-sm text-muted-foreground">Complete control over data and processing</p>
                          </div>
                        </div>
                        <div className="flex items-start gap-3">
                          <Check className="text-green-500 mt-1" size={20} />
                          <div>
                            <h4 className="font-semibold">Regulatory Compliance</h4>
                            <p className="text-sm text-muted-foreground">Built-in compliance with Indian regulations</p>
                          </div>
                        </div>
                        <div className="flex items-start gap-3">
                          <Check className="text-green-500 mt-1" size={20} />
                          <div>
                            <h4 className="font-semibold">Privacy Protection</h4>
                            <p className="text-sm text-muted-foreground">Advanced privacy-preserving techniques</p>
                          </div>
                        </div>
                        <div className="flex items-start gap-3">
                          <Check className="text-green-500 mt-1" size={20} />
                          <div>
                            <h4 className="font-semibold">Audit Trails</h4>
                            <p className="text-sm text-muted-foreground">Comprehensive logging and auditing</p>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-4">
                      <h3 className="text-xl font-semibold text-orange-600">⚡ Performance & Scalability</h3>
                      <div className="space-y-3">
                        <div className="flex items-start gap-3">
                          <Check className="text-green-500 mt-1" size={20} />
                          <div>
                            <h4 className="font-semibold">Optimized Models</h4>
                            <p className="text-sm text-muted-foreground">Efficient models for Indian hardware</p>
                          </div>
                        </div>
                        <div className="flex items-start gap-3">
                          <Check className="text-green-500 mt-1" size={20} />
                          <div>
                            <h4 className="font-semibold">Auto-scaling</h4>
                            <p className="text-sm text-muted-foreground">Dynamic resource allocation</p>
                          </div>
                        </div>
                        <div className="flex items-start gap-3">
                          <Check className="text-green-500 mt-1" size={20} />
                          <div>
                            <h4 className="font-semibold">Load Balancing</h4>
                            <p className="text-sm text-muted-foreground">Intelligent traffic distribution</p>
                          </div>
                        </div>
                        <div className="flex items-start gap-3">
                          <Check className="text-green-500 mt-1" size={20} />
                          <div>
                            <h4 className="font-semibold">Caching Layer</h4>
                            <p className="text-sm text-muted-foreground">Multi-level caching for performance</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="usecases" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-2xl">Real-World Use Cases</CardTitle>
                <CardDescription>Practical applications of BFMF across India</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  <div className="space-y-3 p-4 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-950 dark:to-blue-900 rounded-lg">
                    <div className="flex items-center gap-2">
                      <Building2 className="text-blue-600" size={20} />
                      <h3 className="font-semibold">Government Services</h3>
                    </div>
                    <ul className="text-sm space-y-1 text-muted-foreground">
                      <li>• Automated RTI response generation</li>
                      <li>• Policy analysis and drafting</li>
                      <li>• Government scheme eligibility checker</li>
                      <li>• Compliance audit automation</li>
                    </ul>
                    <Badge variant="secondary">Digital India</Badge>
                  </div>

                  <div className="space-y-3 p-4 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-950 dark:to-green-900 rounded-lg">
                    <div className="flex items-center gap-2">
                      <BookOpen className="text-green-600" size={20} />
                      <h3 className="font-semibold">Education Sector</h3>
                    </div>
                    <ul className="text-sm space-y-1 text-muted-foreground">
                      <li>• AI-powered personalized tutoring</li>
                      <li>• Multilingual educational content</li>
                      <li>• Automated assessment generation</li>
                      <li>• Digital classroom management</li>
                    </ul>
                    <Badge variant="secondary">NEP 2020</Badge>
                  </div>

                  <div className="space-y-3 p-4 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-950 dark:to-purple-900 rounded-lg">
                    <div className="flex items-center gap-2">
                      <Heart className="text-purple-600" size={20} />
                      <h3 className="font-semibold">Healthcare</h3>
                    </div>
                    <ul className="text-sm space-y-1 text-muted-foreground">
                      <li>• Telemedicine consultation support</li>
                      <li>• Multilingual health information</li>
                      <li>• Diagnostic assistance tools</li>
                      <li>• Public health monitoring</li>
                    </ul>
                    <Badge variant="secondary">Ayushman Bharat</Badge>
                  </div>

                  <div className="space-y-3 p-4 bg-gradient-to-br from-yellow-50 to-yellow-100 dark:from-yellow-950 dark:to-yellow-900 rounded-lg">
                    <div className="flex items-center gap-2">
                      <Factory className="text-yellow-600" size={20} />
                      <h3 className="font-semibold">Agriculture</h3>
                    </div>
                    <ul className="text-sm space-y-1 text-muted-foreground">
                      <li>• Crop disease detection</li>
                      <li>• Yield prediction models</li>
                      <li>• Weather-based advisory</li>
                      <li>• Market price intelligence</li>
                    </ul>
                    <Badge variant="secondary">Digital Agriculture</Badge>
                  </div>

                  <div className="space-y-3 p-4 bg-gradient-to-br from-red-50 to-red-100 dark:from-red-950 dark:to-red-900 rounded-lg">
                    <div className="flex items-center gap-2">
                      <TrendingUp className="text-red-600" size={20} />
                      <h3 className="font-semibold">Financial Services</h3>
                    </div>
                    <ul className="text-sm space-y-1 text-muted-foreground">
                      <li>• Fraud detection systems</li>
                      <li>• Credit risk assessment</li>
                      <li>• Transaction monitoring</li>
                      <li>• Regulatory compliance</li>
                    </ul>
                    <Badge variant="secondary">Financial Inclusion</Badge>
                  </div>

                  <div className="space-y-3 p-4 bg-gradient-to-br from-cyan-50 to-cyan-100 dark:from-cyan-950 dark:to-cyan-900 rounded-lg">
                    <div className="flex items-center gap-2">
                      <Users className="text-cyan-600" size={20} />
                      <h3 className="font-semibold">Media & Communication</h3>
                    </div>
                    <ul className="text-sm space-y-1 text-muted-foreground">
                      <li>• Multilingual content generation</li>
                      <li>• Fake news detection</li>
                      <li>• Sentiment analysis</li>
                      <li>• Accessibility tools</li>
                    </ul>
                    <Badge variant="secondary">Media Literacy</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Success Stories</CardTitle>
                <CardDescription>Real implementations of BFMF across India</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-3 p-4 border rounded-lg">
                    <div className="flex items-center gap-2">
                      <Star className="text-yellow-500" size={20} />
                      <h3 className="font-semibold">Karnataka RTI Automation</h3>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Reduced RTI response time from 30 days to 3 days using BFMF's governance AI module, processing over 10,000 requests monthly.
                    </p>
                    <div className="flex gap-2 mt-2">
                      <Badge variant="outline">95% Accuracy</Badge>
                      <Badge variant="outline">90% Time Saved</Badge>
                    </div>
                  </div>

                  <div className="space-y-3 p-4 border rounded-lg">
                    <div className="flex items-center gap-2">
                      <Star className="text-yellow-500" size={20} />
                      <h3 className="font-semibold">Bihar Education Initiative</h3>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Deployed AI teachers across 500 schools, providing personalized learning in 5 local languages, benefiting 50,000+ students.
                    </p>
                    <div className="flex gap-2 mt-2">
                      <Badge variant="outline">40% Improvement</Badge>
                      <Badge variant="outline">5 Languages</Badge>
                    </div>
                  </div>

                  <div className="space-y-3 p-4 border rounded-lg">
                    <div className="flex items-center gap-2">
                      <Star className="text-yellow-500" size={20} />
                      <h3 className="font-semibold">Punjab Agriculture Portal</h3>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Farmers receive crop advisories in Punjabi via SMS, increasing yields by 25% and reducing pesticide use by 30%.
                    </p>
                    <div className="flex gap-2 mt-2">
                      <Badge variant="outline">25% Yield Increase</Badge>
                      <Badge variant="outline">100K Farmers</Badge>
                    </div>
                  </div>

                  <div className="space-y-3 p-4 border rounded-lg">
                    <div className="flex items-center gap-2">
                      <Star className="text-yellow-500" size={20} />
                      <h3 className="font-semibold">Tamil Nadu Healthcare</h3>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Telemedicine platform supporting 8 languages, providing healthcare access to 200+ rural villages with limited medical facilities.
                    </p>
                    <div className="flex gap-2 mt-2">
                      <Badge variant="outline">200+ Villages</Badge>
                      <Badge variant="outline">8 Languages</Badge>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="technical" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-2xl">Technical Architecture</CardTitle>
                <CardDescription>Under the hood of BFMF framework</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-4">
                      <h3 className="text-xl font-semibold text-blue-600">🏗️ Core Components</h3>
                      <div className="space-y-3">
                        <div className="p-3 bg-blue-50 dark:bg-blue-950 rounded-lg">
                          <h4 className="font-semibold">bharat_model/</h4>
                          <p className="text-sm text-muted-foreground">Model architectures and implementations</p>
                          <div className="flex gap-1 mt-1">
                            <Badge variant="outline">LLaMA</Badge>
                            <Badge variant="outline">MoE</Badge>
                            <Badge variant="outline">GLM</Badge>
                          </div>
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-950 rounded-lg">
                          <h4 className="font-semibold">bharat_train/</h4>
                          <p className="text-sm text-muted-foreground">Training infrastructure and tools</p>
                          <div className="flex gap-1 mt-1">
                            <Badge variant="outline">DeepSpeed</Badge>
                            <Badge variant="outline">Trainer</Badge>
                            <Badge variant="outline">Fine-tuning</Badge>
                          </div>
                        </div>
                        <div className="p-3 bg-purple-50 dark:bg-purple-950 rounded-lg">
                          <h4 className="font-semibold">bharat_domains/</h4>
                          <p className="text-sm text-muted-foreground">Domain-specific modules</p>
                          <div className="flex gap-1 mt-1">
                            <Badge variant="outline">Language</Badge>
                            <Badge variant="outline">Governance</Badge>
                            <Badge variant="outline">Education</Badge>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-4">
                      <h3 className="text-xl font-semibold text-green-600">🔧 Supporting Infrastructure</h3>
                      <div className="space-y-3">
                        <div className="p-3 bg-orange-50 dark:bg-orange-950 rounded-lg">
                          <h4 className="font-semibold">bharat_cli/</h4>
                          <p className="text-sm text-muted-foreground">Command-line interface</p>
                          <div className="flex gap-1 mt-1">
                            <Badge variant="outline">Training</Badge>
                            <Badge variant="outline">Deployment</Badge>
                            <Badge variant="outline">Management</Badge>
                          </div>
                        </div>
                        <div className="p-3 bg-red-50 dark:bg-red-950 rounded-lg">
                          <h4 className="font-semibold">bharat_deploy/</h4>
                          <p className="text-sm text-muted-foreground">Deployment and inference</p>
                          <div className="flex gap-1 mt-1">
                            <Badge variant="outline">API Server</Badge>
                            <Badge variant="outline">Inference</Badge>
                            <Badge variant="outline">Scaling</Badge>
                          </div>
                        </div>
                        <div className="p-3 bg-cyan-50 dark:bg-cyan-950 rounded-lg">
                          <h4 className="font-semibold">bharat_registry/</h4>
                          <p className="text-sm text-muted-foreground">Model and data registry</p>
                          <div className="flex gap-1 mt-1">
                            <Badge variant="outline">MLflow</Badge>
                            <Badge variant="outline">Hub Utils</Badge>
                            <Badge variant="outline">Versioning</Badge>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <h3 className="text-xl font-semibold text-purple-600">📊 CLI Commands Examples</h3>
                    <div className="bg-slate-50 dark:bg-slate-900 p-4 rounded-lg">
                      <div className="space-y-2 font-mono text-sm">
                        <div><span className="text-blue-500"># Language AI training</span></div>
                        <div><span className="text-green-600">bharat language train-chatbot</span> --languages hi,en,bn --dataset government_schemes</div>
                        <div className="mt-2"><span className="text-blue-500"># Governance AI deployment</span></div>
                        <div><span className="text-green-600">bharat governance deploy-rti</span> --host 0.0.0.0 --port 8001</div>
                        <div className="mt-2"><span className="text-blue-500"># Education content generation</span></div>
                        <div><span className="text-green-600">bharat education generate-content</span> --model ./models/tutor --topic "Photosynthesis" --subject "Biology"</div>
                        <div className="mt-2"><span className="text-blue-500"># Financial transaction audit</span></div>
                        <div><span className="text-green-600">bharat finance audit-transactions</span> --model ./models/analyst --transactions-file transactions.json</div>
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="text-center p-4 bg-blue-50 dark:bg-blue-950 rounded-lg">
                      <div className="text-2xl font-bold text-blue-600">Python</div>
                      <div className="text-sm text-muted-foreground">Primary Language</div>
                    </div>
                    <div className="text-center p-4 bg-green-50 dark:bg-green-950 rounded-lg">
                      <div className="text-2xl font-bold text-green-600">PyTorch</div>
                      <div className="text-sm text-muted-foreground">ML Framework</div>
                    </div>
                    <div className="text-center p-4 bg-purple-50 dark:bg-purple-950 rounded-lg">
                      <div className="text-2xl font-bold text-purple-600">Open Source</div>
                      <div className="text-sm text-muted-foreground">License</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        {/* Phase 4: Enterprise Features */}
        <Card className="border-2 border-red-200 dark:border-red-800">
          <CardHeader>
            <CardTitle className="text-2xl flex items-center gap-2">
              <span className="text-red-600">Phase 4: Enterprise Features</span>
              <Badge variant="secondary" className="bg-red-100 text-red-800">NEW</Badge>
            </CardTitle>
            <CardDescription>Advanced security and edge AI capabilities for enterprise deployment</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Advanced Security */}
              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-red-100 dark:bg-red-900 rounded-full flex items-center justify-center">
                    <Shield className="text-red-600" size={24} />
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-red-600">🔐 Advanced Security</h3>
                    <p className="text-sm text-muted-foreground">Enterprise-grade privacy and encryption</p>
                  </div>
                </div>
                
                <div className="space-y-3">
                  <div className="p-4 border-l-4 border-red-500 bg-red-50 dark:bg-red-950 rounded-r-lg">
                    <h4 className="font-semibold mb-2">Homomorphic Encryption</h4>
                    <p className="text-sm text-muted-foreground mb-2">
                      Perform computations on encrypted data without decryption, ensuring complete privacy for sensitive information.
                    </p>
                    <div className="flex flex-wrap gap-1">
                      <Badge variant="outline">CKKS Scheme</Badge>
                      <Badge variant="outline">Secure ML</Badge>
                      <Badge variant="outline">Privacy-Preserving</Badge>
                    </div>
                  </div>
                  
                  <div className="p-4 border-l-4 border-orange-500 bg-orange-50 dark:bg-orange-950 rounded-r-lg">
                    <h4 className="font-semibold mb-2">Differential Privacy</h4>
                    <p className="text-sm text-muted-foreground mb-2">
                      Mathematical guarantees of privacy with configurable ε-δ privacy budgets for data analysis and machine learning.
                    </p>
                    <div className="flex flex-wrap gap-1">
                      <Badge variant="outline">Laplace Mechanism</Badge>
                      <Badge variant="outline">Gaussian Mechanism</Badge>
                      <Badge variant="outline">Privacy Accounting</Badge>
                    </div>
                  </div>
                  
                  <div className="p-4 border-l-4 border-yellow-500 bg-yellow-50 dark:bg-yellow-950 rounded-r-lg">
                    <h4 className="font-semibold mb-2">Secure Federated Learning</h4>
                    <p className="text-sm text-muted-foreground mb-2">
                      Train models across distributed devices while keeping data local and updates encrypted.
                    </p>
                    <div className="flex flex-wrap gap-1">
                      <Badge variant="outline">Encrypted Aggregation</Badge>
                      <Badge variant="outline">Cross-Device</Badge>
                      <Badge variant="outline">Zero-Knowledge</Badge>
                    </div>
                  </div>
                </div>
              </div>

              {/* Edge AI */}
              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center">
                    <Cpu className="text-blue-600" size={24} />
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-blue-600">📱 Edge AI</h3>
                    <p className="text-sm text-muted-foreground">On-device inference for real-time applications</p>
                  </div>
                </div>
                
                <div className="space-y-3">
                  <div className="p-4 border-l-4 border-blue-500 bg-blue-50 dark:bg-blue-950 rounded-r-lg">
                    <h4 className="font-semibold mb-2">Model Optimization</h4>
                    <p className="text-sm text-muted-foreground mb-2">
                      Advanced optimization techniques including quantization, pruning, and knowledge distillation for edge deployment.
                    </p>
                    <div className="flex flex-wrap gap-1">
                      <Badge variant="outline">8-bit Quantization</Badge>
                      <Badge variant="outline">60% Pruning</Badge>
                      <Badge variant="outline">Model Compression</Badge>
                    </div>
                  </div>
                  
                  <div className="p-4 border-l-4 border-cyan-500 bg-cyan-50 dark:bg-cyan-950 rounded-r-lg">
                    <h4 className="font-semibold mb-2">Edge Inference Engine</h4>
                    <p className="text-sm text-muted-foreground mb-2">
                      High-performance inference engine optimized for mobile, embedded, and IoT devices with minimal resource usage.
                    </p>
                    <div className="flex flex-wrap gap-1">
                      <Badge variant="outline">MobileNet</Badge>
                      <Badge variant="outline">TinyML</Badge>
                      <Badge variant="outline">Real-time</Badge>
                    </div>
                  </div>
                  
                  <div className="p-4 border-l-4 border-green-500 bg-green-50 dark:bg-green-950 rounded-r-lg">
                    <h4 className="font-semibold mb-2">Device Management</h4>
                    <p className="text-sm text-muted-foreground mb-2">
                      Complete lifecycle management for edge models including deployment, monitoring, and rollback capabilities.
                    </p>
                    <div className="flex flex-wrap gap-1">
                      <Badge variant="outline">Model Registry</Badge>
                      <Badge variant="outline">Version Control</Badge>
                      <Badge variant="outline">Performance Monitoring</Badge>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Enterprise Integration */}
            <div className="mt-8 space-y-4">
              <h3 className="text-xl font-semibold text-purple-600 flex items-center gap-2">
                <Building2 size={20} />
                Enterprise Integration Capabilities
              </h3>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-950 dark:to-purple-900 rounded-lg">
                  <h4 className="font-semibold mb-2">Healthcare</h4>
                  <p className="text-sm text-muted-foreground">
                    Secure patient monitoring with encrypted health data and real-time anomaly detection on edge devices.
                  </p>
                  <div className="mt-2">
                    <Badge variant="secondary">HIPAA Compliant</Badge>
                  </div>
                </div>
                
                <div className="p-4 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-950 dark:to-blue-900 rounded-lg">
                  <h4 className="font-semibold mb-2">Finance</h4>
                  <p className="text-sm text-muted-foreground">
                    Privacy-preserving fraud detection and risk analysis with homomorphic encryption on financial data.
                  </p>
                  <div className="mt-2">
                    <Badge variant="secondary">RBI Compliant</Badge>
                  </div>
                </div>
                
                <div className="p-4 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-950 dark:to-green-900 rounded-lg">
                  <h4 className="font-semibold mb-2">Government</h4>
                  <p className="text-sm text-muted-foreground">
                    Secure citizen services with differential privacy and edge AI for offline capability in rural areas.
                  </p>
                  <div className="mt-2">
                    <Badge variant="secondary">Digital India</Badge>
                  </div>
                </div>
              </div>
            </div>

            {/* Performance Metrics */}
            <div className="mt-8 grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="text-center p-4 bg-red-50 dark:bg-red-950 rounded-lg">
                <div className="text-2xl font-bold text-red-600">100%</div>
                <div className="text-sm text-muted-foreground">Data Privacy</div>
              </div>
              <div className="text-center p-4 bg-blue-50 dark:bg-blue-950 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">&lt;25ms</div>
                <div className="text-sm text-muted-foreground">Edge Inference</div>
              </div>
              <div className="text-center p-4 bg-green-50 dark:bg-green-950 rounded-lg">
                <div className="text-2xl font-bold text-green-600">75%</div>
                <div className="text-sm text-muted-foreground">Size Reduction</div>
              </div>
              <div className="text-center p-4 bg-purple-50 dark:bg-purple-950 rounded-lg">
                <div className="text-2xl font-bold text-purple-600">Enterprise</div>
                <div className="text-sm text-muted-foreground">Ready</div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Phase 5: System Intelligence */}
        <Card className="border-2 border-indigo-200 dark:border-indigo-800">
          <CardHeader>
            <CardTitle className="text-2xl flex items-center gap-2">
              <span className="text-indigo-600">Phase 5: System Intelligence</span>
              <Badge variant="secondary" className="bg-indigo-100 text-indigo-800">NEW</Badge>
            </CardTitle>
            <CardDescription>AutoML pipeline and multi-agent system for intelligent automation</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* AutoML Pipeline */}
              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-indigo-100 dark:bg-indigo-900 rounded-full flex items-center justify-center">
                    <Zap className="text-indigo-600" size={24} />
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-indigo-600">🤖 AutoML Pipeline</h3>
                    <p className="text-sm text-muted-foreground">Automated machine learning with intelligent optimization</p>
                  </div>
                </div>
                
                <div className="space-y-3">
                  <div className="p-4 border-l-4 border-indigo-500 bg-indigo-50 dark:bg-indigo-950 rounded-r-lg">
                    <h4 className="font-semibold mb-2">Automated Model Selection</h4>
                    <p className="text-sm text-muted-foreground mb-2">
                      Intelligent model selection from linear, tree-based, neural network, and ensemble methods with automatic hyperparameter tuning.
                    </p>
                    <div className="flex flex-wrap gap-1">
                      <Badge variant="outline">Auto Selection</Badge>
                      <Badge variant="outline">Hyperparameter Tuning</Badge>
                      <Badge variant="outline">Cross-Validation</Badge>
                    </div>
                  </div>
                  
                  <div className="p-4 border-l-4 border-purple-500 bg-purple-50 dark:bg-purple-950 rounded-r-lg">
                    <h4 className="font-semibold mb-2">Feature Engineering</h4>
                    <p className="text-sm text-muted-foreground mb-2">
                      Automated feature engineering including polynomial features, interactions, statistical features, and intelligent encoding.
                    </p>
                    <div className="flex flex-wrap gap-1">
                      <Badge variant="outline">Auto Features</Badge>
                      <Badge variant="outline">Encoding</Badge>
                      <Badge variant="outline">Scaling</Badge>
                    </div>
                  </div>
                  
                  <div className="p-4 border-l-4 border-pink-500 bg-pink-50 dark:bg-pink-950 rounded-r-lg">
                    <h4 className="font-semibold mb-2">Ensemble Building</h4>
                    <p className="text-sm text-muted-foreground mb-2">
                      Automatic ensemble construction combining best-performing models for optimal predictive accuracy.
                    </p>
                    <div className="flex flex-wrap gap-1">
                      <Badge variant="outline">Voting</Badge>
                      <Badge variant="outline">Stacking</Badge>
                      <Badge variant="outline">Blending</Badge>
                    </div>
                  </div>
                </div>
              </div>

              {/* Multi-Agent System */}
              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-teal-100 dark:bg-teal-900 rounded-full flex items-center justify-center">
                    <Users className="text-teal-600" size={24} />
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-teal-600">👥 Multi-Agent System</h3>
                    <p className="text-sm text-muted-foreground">Collaborative AI agents for problem-solving</p>
                  </div>
                </div>
                
                <div className="space-y-3">
                  <div className="p-4 border-l-4 border-teal-500 bg-teal-50 dark:bg-teal-950 rounded-r-lg">
                    <h4 className="font-semibold mb-2">Specialist Agents</h4>
                    <p className="text-sm text-muted-foreground mb-2">
                      Domain-specific agents with expertise in analysis, optimization, prediction, and validation working collaboratively.
                    </p>
                    <div className="flex flex-wrap gap-1">
                      <Badge variant="outline">Data Analyst</Badge>
                      <Badge variant="outline">Optimizer</Badge>
                      <Badge variant="outline">Predictor</Badge>
                    </div>
                  </div>
                  
                  <div className="p-4 border-l-4 border-cyan-500 bg-cyan-50 dark:bg-cyan-950 rounded-r-lg">
                    <h4 className="font-semibold mb-2">Intelligent Coordination</h4>
                    <p className="text-sm text-muted-foreground mb-2">
                      Smart task distribution, load balancing, and dependency management with real-time performance monitoring.
                    </p>
                    <div className="flex flex-wrap gap-1">
                      <Badge variant="outline">Task Scheduling</Badge>
                      <Badge variant="outline">Load Balancing</Badge>
                      <Badge variant="outline">Resource Management</Badge>
                    </div>
                  </div>
                  
                  <div className="p-4 border-l-4 border-emerald-500 bg-emerald-50 dark:bg-emerald-950 rounded-r-lg">
                    <h4 className="font-semibold mb-2">Collective Learning</h4>
                    <p className="text-sm text-muted-foreground mb-2">
                      Agents learn from each other's experiences through reinforcement, imitation, and collaborative learning strategies.
                    </p>
                    <div className="flex flex-wrap gap-1">
                      <Badge variant="outline">Knowledge Sharing</Badge>
                      <Badge variant="outline">Experience Transfer</Badge>
                      <Badge variant="outline">Adaptive Learning</Badge>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Intelligent Integration */}
            <div className="mt-8 space-y-4">
              <h3 className="text-xl font-semibold text-purple-600 flex items-center gap-2">
                <Cpu size={20} />
                Intelligent System Integration
              </h3>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-4 bg-gradient-to-br from-indigo-50 to-indigo-100 dark:from-indigo-950 dark:to-indigo-900 rounded-lg">
                  <h4 className="font-semibold mb-2">Smart Automation</h4>
                  <p className="text-sm text-muted-foreground">
                    End-to-end automation from data preprocessing to model deployment with intelligent decision-making.
                  </p>
                  <div className="mt-2">
                    <Badge variant="secondary">Zero-Code ML</Badge>
                  </div>
                </div>
                
                <div className="p-4 bg-gradient-to-br from-teal-50 to-teal-100 dark:from-teal-950 dark:to-teal-900 rounded-lg">
                  <h4 className="font-semibold mb-2">Adaptive Intelligence</h4>
                  <p className="text-sm text-muted-foreground">
                    Self-improving systems that learn from experience and adapt to new data patterns automatically.
                  </p>
                  <div className="mt-2">
                    <Badge variant="secondary">Self-Learning</Badge>
                  </div>
                </div>
                
                <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-950 dark:to-purple-900 rounded-lg">
                  <h4 className="font-semibold mb-2">Collaborative AI</h4>
                  <p className="text-sm text-muted-foreground">
                    Multiple AI agents working together to solve complex problems that single systems cannot handle alone.
                  </p>
                  <div className="mt-2">
                    <Badge variant="secondary">Team Intelligence</Badge>
                  </div>
                </div>
              </div>
            </div>

            {/* System Intelligence Workflow */}
            <div className="mt-8 space-y-4">
              <h3 className="text-xl font-semibold text-blue-600 flex items-center gap-2">
                <TrendingUp size={20} />
                System Intelligence Workflow
              </h3>
              
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="text-center p-4 bg-blue-50 dark:bg-blue-950 rounded-lg">
                  <div className="w-8 h-8 bg-blue-200 dark:bg-blue-800 rounded-full flex items-center justify-center mx-auto mb-2">
                    <span className="font-bold text-blue-600 text-sm">1</span>
                  </div>
                  <h4 className="font-semibold text-sm mb-1">Data Ingestion</h4>
                  <p className="text-xs text-muted-foreground">Automated data collection and preprocessing</p>
                </div>
                
                <div className="text-center p-4 bg-indigo-50 dark:bg-indigo-950 rounded-lg">
                  <div className="w-8 h-8 bg-indigo-200 dark:bg-indigo-800 rounded-full flex items-center justify-center mx-auto mb-2">
                    <span className="font-bold text-indigo-600 text-sm">2</span>
                  </div>
                  <h4 className="font-semibold text-sm mb-1">AutoML Processing</h4>
                  <p className="text-xs text-muted-foreground">Intelligent model selection and training</p>
                </div>
                
                <div className="text-center p-4 bg-teal-50 dark:bg-teal-950 rounded-lg">
                  <div className="w-8 h-8 bg-teal-200 dark:bg-teal-800 rounded-full flex items-center justify-center mx-auto mb-2">
                    <span className="font-bold text-teal-600 text-sm">3</span>
                  </div>
                  <h4 className="font-semibold text-sm mb-1">Multi-Agent Analysis</h4>
                  <p className="text-xs text-muted-foreground">Collaborative problem-solving and validation</p>
                </div>
                
                <div className="text-center p-4 bg-purple-50 dark:bg-purple-950 rounded-lg">
                  <div className="w-8 h-8 bg-purple-200 dark:bg-purple-800 rounded-full flex items-center justify-center mx-auto mb-2">
                    <span className="font-bold text-purple-600 text-sm">4</span>
                  </div>
                  <h4 className="font-semibold text-sm mb-1">Intelligent Deployment</h4>
                  <p className="text-xs text-muted-foreground">Optimized deployment with continuous learning</p>
                </div>
              </div>
            </div>

            {/* Performance Metrics */}
            <div className="mt-8 grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="text-center p-4 bg-indigo-50 dark:bg-indigo-950 rounded-lg">
                <div className="text-2xl font-bold text-indigo-600">95%</div>
                <div className="text-sm text-muted-foreground">Automation Rate</div>
              </div>
              <div className="text-center p-4 bg-teal-50 dark:bg-teal-950 rounded-lg">
                <div className="text-2xl font-bold text-teal-600">10x</div>
                <div className="text-sm text-muted-foreground">Faster Development</div>
              </div>
              <div className="text-center p-4 bg-purple-50 dark:bg-purple-950 rounded-lg">
                <div className="text-2xl font-bold text-purple-600">85%</div>
                <div className="text-sm text-muted-foreground">Resource Efficiency</div>
              </div>
              <div className="text-center p-4 bg-pink-50 dark:bg-pink-950 rounded-lg">
                <div className="text-2xl font-bold text-pink-600">AI</div>
                <div className="text-sm text-muted-foreground">First Platform</div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Getting Started */}
        <Card className="border-2 border-green-200 dark:border-green-800">
          <CardHeader>
            <CardTitle className="text-2xl text-green-600">Getting Started with BFMF</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center">
                    <span className="font-bold text-blue-600">1</span>
                  </div>
                  <h3 className="text-lg font-semibold">Installation</h3>
                </div>
                <p className="text-sm text-muted-foreground">
                  Install BFMF with a single command and get access to all 10 domain modules and tools.
                </p>
                <div className="bg-slate-100 dark:bg-slate-800 p-2 rounded text-sm font-mono">
                  pip install bharat-fm
                </div>
              </div>

              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 bg-green-100 dark:bg-green-900 rounded-full flex items-center justify-center">
                    <span className="font-bold text-green-600">2</span>
                  </div>
                  <h3 className="text-lg font-semibold">Choose Domain</h3>
                </div>
                <p className="text-sm text-muted-foreground">
                  Select from 10 domain-specific modules based on your use case requirements.
                </p>
                <div className="flex flex-wrap gap-1">
                  <Badge variant="outline">Language</Badge>
                  <Badge variant="outline">Governance</Badge>
                  <Badge variant="outline">Education</Badge>
                  <Badge variant="outline">+7 more</Badge>
                </div>
              </div>

              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 bg-purple-100 dark:bg-purple-900 rounded-full flex items-center justify-center">
                    <span className="font-bold text-purple-600">3</span>
                  </div>
                  <h3 className="text-lg font-semibold">Deploy & Scale</h3>
                </div>
                <p className="text-sm text-muted-foreground">
                  Use the CLI to train, fine-tune, and deploy your AI application with one command.
                </p>
                <div className="bg-slate-100 dark:bg-slate-800 p-2 rounded text-sm font-mono">
                  bharat deploy --domain governance
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Conclusion */}
        <Card className="border-2 border-purple-200 dark:border-purple-800">
          <CardHeader>
            <CardTitle className="text-2xl text-purple-600">Why BFMF?</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-lg leading-relaxed">
              The Bharat Foundation Model Framework is more than just an AI framework—it's <strong>India's answer to sovereign AI</strong>. Built by Indians, for India, BFMF provides the tools, capabilities, and infrastructure needed to power India's digital transformation across all sectors.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
              <div className="space-y-3">
                <h3 className="text-lg font-semibold text-blue-600">🇮🇳 For India</h3>
                <ul className="space-y-2 text-sm">
                  <li className="flex items-center gap-2">
                    <Check className="text-green-500" size={16} />
                    <span>Native support for 22+ Indian languages</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <Check className="text-green-500" size={16} />
                    <span>Cultural context understanding</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <Check className="text-green-500" size={16} />
                    <span>Compliance with Indian regulations</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <Check className="text-green-500" size={16} />
                    <span>Domain-specific for Indian sectors</span>
                  </li>
                </ul>
              </div>
              <div className="space-y-3">
                <h3 className="text-lg font-semibold text-green-600">🚀 For Innovation</h3>
                <ul className="space-y-2 text-sm">
                  <li className="flex items-center gap-2">
                    <Check className="text-green-500" size={16} />
                    <span>Open source and extensible</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <Check className="text-green-500" size={16} />
                    <span>Production-ready from day one</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <Check className="text-green-500" size={16} />
                    <span>Comprehensive developer tools</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <Check className="text-green-500" size={16} />
                    <span>Growing community and support</span>
                  </li>
                </ul>
              </div>
            </div>
            <div className="bg-purple-50 dark:bg-purple-950 p-4 rounded-lg mt-6">
              <h4 className="font-semibold text-purple-600 mb-2">Join the Movement:</h4>
              <p className="text-sm">
                BFMF is empowering organizations across India to build AI applications that are truly Indian in context, language, and infrastructure. Join us in shaping India's AI future.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}